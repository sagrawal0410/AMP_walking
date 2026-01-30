import numpy as np
import os
import yaml

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import class_to_dict
from isaaclab.utils.string import resolve_matching_names


def format_value(x):
    """Format values for YAML output, converting tuples to lists and handling special types."""
    # Handle None first
    if x is None:
        return None
    
    # Handle slice objects (can't be serialized)
    if isinstance(x, slice):
        return None
    
    # Handle config objects (like SceneEntityCfg) that need to be converted to dict first
    if hasattr(x, 'to_dict') and not isinstance(x, (dict, list, tuple, str, int, float, bool, np.ndarray, np.integer, np.floating)):
        try:
            return format_value(x.to_dict())
        except:
            pass
    
    # Try class_to_dict for objects that can't be serialized
    try:
        if not isinstance(x, (dict, list, tuple, str, int, float, bool, np.ndarray, np.integer, np.floating)):
            converted = class_to_dict(x)
            if converted != x:  # Only use if conversion actually changed something
                return format_value(converted)
    except:
        pass
    
    if isinstance(x, float):
        return float(f"{x:.3g}")
    elif isinstance(x, (tuple, list)):
        # Convert tuples to lists to avoid !!python/tuple tags in YAML
        return [format_value(i) for i in x]
    elif isinstance(x, dict):
        return {k: format_value(v) for k, v in x.items()}
    elif isinstance(x, np.ndarray):
        return format_value(x.tolist())
    elif isinstance(x, np.integer):
        return int(x)
    elif isinstance(x, np.floating):
        return float(x)
    else:
        # For any other type, try to convert or return None
        try:
            if hasattr(x, '__dict__'):
                return format_value(class_to_dict(x))
        except:
            pass
        return None  # Can't serialize, return None


def export_deploy_cfg(env: ManagerBasedRLEnv, log_dir):
    asset: Articulation = env.scene["robot"]
    joint_sdk_names = env.cfg.scene.robot.joint_sdk_names
    print(f"Internal joint names ({len(asset.data.joint_names)}):")
    for i, name in enumerate(asset.data.joint_names):
        print(f"  [{i:2d}] {name}")

    print(f"\nSDK joint names ({len(joint_sdk_names)}):")
    for i, name in enumerate(joint_sdk_names):
        print(f"  [{i:2d}] {name}")
    joint_ids_map, _ = resolve_matching_names(asset.data.joint_names, joint_sdk_names, preserve_order=True)
    print(f"\njoint_ids_map result: {joint_ids_map}")
    cfg = {}  # noqa: SIM904
    # Convert joint_ids_map to list of integers (resolve_matching_names returns numpy array or list)
    if isinstance(joint_ids_map, np.ndarray):
        joint_ids_map = joint_ids_map.tolist()
    
    # Validate joint_ids_map is not empty
    if not joint_ids_map or len(joint_ids_map) == 0:
        raise ValueError(
            f"joint_ids_map is empty! This means joint name matching failed.\n"
            f"Lab joint names: {asset.data.joint_names}\n"
            f"SDK joint names: {joint_sdk_names}\n"
            f"Check that joint_sdk_names in robot config matches the actual joint names."
        )
    
    # Ensure all values are integers, not None
    cfg["joint_ids_map"] = [int(x) if x is not None and not np.isnan(x) else 0 for x in joint_ids_map]
    
    # Validate the result is not empty
    if len(cfg["joint_ids_map"]) == 0:
        raise ValueError("joint_ids_map became empty after conversion! Check joint name matching.")
    cfg["step_dt"] = env.cfg.sim.dt * env.cfg.decimation
    stiffness = np.zeros(len(joint_sdk_names))
    stiffness[joint_ids_map] = asset.data.default_joint_stiffness[0].detach().cpu().numpy().tolist()
    cfg["stiffness"] = stiffness.tolist()
    damping = np.zeros(len(joint_sdk_names))
    damping[joint_ids_map] = asset.data.default_joint_damping[0].detach().cpu().numpy().tolist()
    cfg["damping"] = damping.tolist()
    # default_joint_pos should be in SDK joint order (same as stiffness/damping)
    default_joint_pos = np.zeros(len(joint_sdk_names))
    default_joint_pos[joint_ids_map] = asset.data.default_joint_pos[0].detach().cpu().numpy().tolist()
    cfg["default_joint_pos"] = default_joint_pos.tolist()

    # --- commands ---
    cfg["commands"] = {}
    if hasattr(env.cfg.commands, "base_velocity"):  # some environments do not have base_velocity command
        cfg["commands"]["base_velocity"] = {}
        if hasattr(env.cfg.commands.base_velocity, "limit_ranges"):
            ranges = env.cfg.commands.base_velocity.limit_ranges.to_dict()
        else:
            ranges = env.cfg.commands.base_velocity.ranges.to_dict()
        for item_name in ["lin_vel_x", "lin_vel_y", "ang_vel_z"]:
            ranges[item_name] = list(ranges[item_name])
        # Handle heading - convert tuple to list or set to null
        if "heading" in ranges:
            if ranges["heading"] is None:
                ranges["heading"] = None
            elif isinstance(ranges["heading"], (tuple, list)):
                # Convert tuple/list to list, but if it's a range tuple, keep as list
                ranges["heading"] = list(ranges["heading"]) if len(ranges["heading"]) > 0 else None
            # If it's already a list, keep it as is
        else:
            ranges["heading"] = None
        cfg["commands"]["base_velocity"]["ranges"] = ranges

    # --- actions ---
    action_names = env.action_manager.active_terms
    action_terms = zip(action_names, env.action_manager._terms.values())
    cfg["actions"] = {}
    for action_name, action_term in action_terms:
        term_cfg = action_term.cfg.copy()
        
        # Get the class name from class_type before converting to dict
        action_class_name = term_cfg.class_type.__name__
        
        if isinstance(term_cfg.scale, float):
            term_cfg.scale = [term_cfg.scale for _ in range(action_term.action_dim)]
        else:  # dict
            term_cfg.scale = action_term._scale[0].detach().cpu().numpy().tolist()

        if term_cfg.clip is not None:
            term_cfg.clip = action_term._clip[0].detach().cpu().numpy().tolist()

        # Handle offset for JointPositionAction and JointVelocityAction
        if action_class_name in ["JointPositionAction", "JointVelocityAction"]:
            if term_cfg.use_default_offset:
                term_cfg.offset = action_term._offset[0].detach().cpu().numpy().tolist()
            else:
                term_cfg.offset = [0.0 for _ in range(action_term.action_dim)]

        # clean cfg
        term_cfg = term_cfg.to_dict()

        for _ in ["class_type", "asset_name", "debug_vis", "preserve_order", "use_default_offset"]:
            if _ in term_cfg:
                del term_cfg[_]
        
        # Ensure offset is always a list (not a scalar) if it exists and is not None
        # The deploy code expects offset to be either null or a vector
        if "offset" in term_cfg:
            if term_cfg["offset"] is None:
                # Keep as None (will be written as null in YAML)
                pass
            elif isinstance(term_cfg["offset"], (int, float)):
                # Convert scalar to list
                term_cfg["offset"] = [float(term_cfg["offset"]) for _ in range(action_term.action_dim)]
            elif not isinstance(term_cfg["offset"], list):
                # Convert other types to list if needed
                term_cfg["offset"] = list(term_cfg["offset"])
        
        # Use class name instead of term name for deploy compatibility
        cfg["actions"][action_class_name] = term_cfg

        if action_term._joint_ids == slice(None):
            cfg["actions"][action_class_name]["joint_ids"] = None
        else:
            cfg["actions"][action_class_name]["joint_ids"] = action_term._joint_ids
        
        # Remove joint_names entirely - C++ doesn't need it, and it causes parsing errors if it contains None
        if "joint_names" in cfg["actions"][action_class_name]:
            del cfg["actions"][action_class_name]["joint_names"]
        
        # Remove clip if it's None (C++ handles missing clip gracefully)
        if "clip" in cfg["actions"][action_class_name] and cfg["actions"][action_class_name]["clip"] is None:
            del cfg["actions"][action_class_name]["clip"]

    # --- observations ---
    # List of observations registered in the deploy code
    # These are the only observations that can be used in deploy.yaml
    registered_observations = {
        "base_ang_vel",
        "projected_gravity",
        "joint_pos",
        "joint_pos_rel",
        "joint_vel",
        "joint_vel_rel",
        "last_action",
        "velocity_commands",
        "gait_phase",
        "keyboard_velocity_commands",  # Robot-specific, registered in State_RLBase.cpp
        "root_local_rot_tan_norm",  # AMP-specific
        "key_body_pos_b",  # AMP-specific
    }
    
    # Check if this is an AMP policy by looking for AMP-specific function names
    # We need to check function names, not observation names, because the mapping happens later
    has_amp_terms = False
    for obs_name, obs_cfg in zip(env.observation_manager.active_terms["policy"], 
                                  env.observation_manager._group_obs_term_cfgs["policy"]):
        func_name = obs_cfg.func.__name__ if hasattr(obs_cfg.func, '__name__') else str(obs_cfg.func)
        if func_name in ["root_local_rot_tan_norm", "key_body_pos_b"]:
            has_amp_terms = True
            break
    
    # Map from training observation names to deploy observation names
    # This is needed because training uses different names than deploy function names
    obs_name_mapping = {
        "actions": "last_action",  # training uses "actions", deploy uses "last_action"
    }
    
    # For AMP policies, map relative joint terms to absolute terms
    if has_amp_terms:
        obs_name_mapping["joint_pos_rel"] = "joint_pos"
        obs_name_mapping["joint_vel_rel"] = "joint_vel"
    
    obs_names = env.observation_manager.active_terms["policy"]
    obs_cfgs = env.observation_manager._group_obs_term_cfgs["policy"]
    obs_terms = zip(obs_names, obs_cfgs)
    
    # Store observation order explicitly for AMP policies
    obs_order = []
    cfg["observations"] = {}
    
    for obs_name, obs_cfg in obs_terms:
        # Get the function name to determine the deploy observation name
        func_name = obs_cfg.func.__name__ if hasattr(obs_cfg.func, '__name__') else str(obs_cfg.func)
        
        # Determine deploy observation name
        # Priority: 1) function name if registered (most accurate), 2) explicit mapping, 3) original name if registered
        deploy_obs_name = None
        
        # Check if function name is registered first (most accurate mapping)
        # This handles cases where training name differs from function name
        if func_name in registered_observations:
            deploy_obs_name = func_name
        # Check explicit mapping second (for actions -> last_action)
        elif obs_name in obs_name_mapping:
            deploy_obs_name = obs_name_mapping[obs_name]
        # Check if original name is registered
        elif obs_name in registered_observations:
            deploy_obs_name = obs_name
        # Special case: generated_commands -> keyboard_velocity_commands for AMP, velocity_commands for velocity
        elif func_name == "generated_commands":
            # For AMP policies, always use keyboard_velocity_commands
            # Use the has_amp_terms variable already computed above
            if has_amp_terms and "keyboard_velocity_commands" in registered_observations:
                deploy_obs_name = "keyboard_velocity_commands"
            elif "velocity_commands" in registered_observations:
                deploy_obs_name = "velocity_commands"
        
        # Skip if no valid deploy observation name found
        if deploy_obs_name is None:
            print(f"[WARNING] Skipping observation '{obs_name}' (func: {func_name}) - not registered in deploy code")
            continue
        
        # Special handling for keyboard_velocity_commands - use velocity_commands params
        if deploy_obs_name == "keyboard_velocity_commands":
            # keyboard_velocity_commands uses the same params as velocity_commands
            # but is registered separately in State_RLBase.cpp
            # Ensure params has command_name set
            if "params" not in term_cfg or term_cfg.get("params") is None:
                term_cfg["params"] = {}
            if not isinstance(term_cfg["params"], dict):
                term_cfg["params"] = {}
            if "command_name" not in term_cfg["params"] or term_cfg["params"].get("command_name") is None:
                term_cfg["params"]["command_name"] = "base_velocity"
            
        obs_dims = tuple(obs_cfg.func(env, **obs_cfg.params).shape)
        term_cfg = obs_cfg.copy()
        
        # Handle scale - extract from config properly (matching export_deploy_cfg_other.py)
        if term_cfg.scale is not None:
            # Try to detach if it's a tensor, otherwise use as-is
            try:
                if hasattr(term_cfg.scale, 'detach'):
                    scale = term_cfg.scale.detach().cpu().numpy().tolist()
                elif isinstance(term_cfg.scale, (list, tuple, np.ndarray)):
                    scale = list(term_cfg.scale) if not isinstance(term_cfg.scale, np.ndarray) else term_cfg.scale.tolist()
                else:
                    scale = term_cfg.scale
            except:
                # Fallback: try direct conversion
                scale = term_cfg.scale
            
            if isinstance(scale, float):
                term_cfg.scale = [scale for _ in range(obs_dims[1])]
            else:
                term_cfg.scale = scale
        else:
            term_cfg.scale = [1.0 for _ in range(obs_dims[1])]
        
        # Handle clip - extract from config properly
        if term_cfg.clip is not None:
            try:
                if hasattr(term_cfg.clip, 'detach'):
                    term_cfg.clip = term_cfg.clip.detach().cpu().numpy().tolist()
                elif isinstance(term_cfg.clip, (list, tuple, np.ndarray)):
                    term_cfg.clip = list(term_cfg.clip) if not isinstance(term_cfg.clip, np.ndarray) else term_cfg.clip.tolist()
                else:
                    term_cfg.clip = list(term_cfg.clip)
            except:
                # Fallback: try direct conversion
                term_cfg.clip = list(term_cfg.clip) if hasattr(term_cfg.clip, '__iter__') else None
        
        # Ensure history_length is set (default to 5 for AMP policies, 1 otherwise)
        if not hasattr(term_cfg, 'history_length') or term_cfg.history_length is None or term_cfg.history_length == 0:
            # For AMP policies, use history_length=5, otherwise use 1
            term_cfg.history_length = 5 if has_amp_terms else 1

        # clean cfg
        term_cfg = term_cfg.to_dict()
        # Ensure history_length is in the dict and not None (must be an integer)
        if "history_length" not in term_cfg or term_cfg.get("history_length") is None:
            term_cfg["history_length"] = 5 if has_amp_terms else 1
        else:
            # Ensure it's an int, not None
            term_cfg["history_length"] = int(term_cfg["history_length"]) if term_cfg["history_length"] is not None else (5 if has_amp_terms else 1)
        for _ in ["func", "modifiers", "noise", "flatten_history_dim"]:
            if _ in term_cfg:
                del term_cfg[_]
        
        # Ensure params is a dict (not None or missing)
        # Params should come from obs_cfg.params, not term_cfg
        # Need to recursively convert config objects (like SceneEntityCfg) to dicts
        def convert_params_to_dict(params_obj):
            """Recursively convert params object to dict, handling config objects."""
            if params_obj is None:
                return {}
            
            # Handle slice objects (can't be serialized) - skip them
            if isinstance(params_obj, slice):
                return None  # Slice objects can't be serialized, C++ doesn't need them
            
            # If it's already a dict, recursively process it and filter out None values
            if isinstance(params_obj, dict):
                result = {}
                for k, v in params_obj.items():
                    converted_v = convert_params_to_dict(v)
                    # Skip None values and slices
                    if converted_v is not None and not isinstance(converted_v, slice):
                        result[k] = converted_v
                return result
            
            # If it's a config object with to_dict method, use it
            if hasattr(params_obj, 'to_dict'):
                return convert_params_to_dict(params_obj.to_dict())
            
            # If it's a list/tuple, recursively process elements and filter out None/slice
            if isinstance(params_obj, (list, tuple)):
                result = []
                for item in params_obj:
                    if isinstance(item, slice):
                        continue  # Skip slice objects
                    converted = convert_params_to_dict(item)
                    if converted is not None:  # Only add non-None values
                        result.append(converted)
                return result
            
            # For primitive types or objects we can't convert, try class_to_dict
            try:
                converted = class_to_dict(params_obj)
                if isinstance(converted, dict):
                    # Filter out slice objects and other non-serializable items
                    filtered = {}
                    for k, v in converted.items():
                        if not isinstance(v, slice):
                            converted_v = convert_params_to_dict(v)
                            if converted_v is not None:
                                filtered[k] = converted_v
                    return filtered if filtered else convert_params_to_dict(converted)
            except:
                pass
            
            # Fallback: return as-is if it's a primitive type
            if isinstance(params_obj, (str, int, float, bool, type(None))):
                return params_obj
            
            # Last resort: try to convert to dict
            try:
                if hasattr(params_obj, '__iter__') and not isinstance(params_obj, str):
                    return dict((k, convert_params_to_dict(v)) for k, v in params_obj.items() if not isinstance(v, slice))
            except:
                pass
            
            # If all else fails, return empty dict
            return {}
        
        if hasattr(obs_cfg, 'params') and obs_cfg.params is not None:
            term_cfg["params"] = convert_params_to_dict(obs_cfg.params)
        elif "params" not in term_cfg or term_cfg.get("params") is None:
            term_cfg["params"] = {}
        elif not isinstance(term_cfg["params"], dict):
            term_cfg["params"] = convert_params_to_dict(term_cfg["params"])
        
        # Ensure params is always a dict (never None)
        if term_cfg.get("params") is None:
            term_cfg["params"] = {}
        
        # Post-process params to ensure no None values in critical fields
        if isinstance(term_cfg.get("params"), dict):
            params_dict = term_cfg["params"]
            # Fix command_name for velocity_commands/keyboard_velocity_commands
            if deploy_obs_name in ["velocity_commands", "keyboard_velocity_commands"]:
                # Always set command_name - it's required for C++ parsing
                params_dict["command_name"] = "base_velocity"
        
        # Special handling for key_body_pos_b: ensure body_names are exported correctly
        if deploy_obs_name == "key_body_pos_b" and "params" in term_cfg:
            # Extract body_names from obs_cfg.params (source of truth)
            params_dict = term_cfg["params"]
            
            # Try to get body_names from the original obs_cfg.params first
            body_names = None
            if hasattr(obs_cfg, 'params') and obs_cfg.params is not None:
                try:
                    if hasattr(obs_cfg.params, 'get'):
                        asset_cfg = obs_cfg.params.get('asset_cfg', None)
                        if asset_cfg is not None:
                            if hasattr(asset_cfg, 'body_names'):
                                body_names = asset_cfg.body_names
                            elif isinstance(asset_cfg, dict) and 'body_names' in asset_cfg:
                                body_names = asset_cfg['body_names']
                except:
                    pass
            
            # If not found, try from params_dict
            if body_names is None:
                if isinstance(params_dict, dict) and "asset_cfg" in params_dict:
                    asset_cfg_dict = params_dict["asset_cfg"]
                    if isinstance(asset_cfg_dict, dict):
                        body_names = asset_cfg_dict.get("body_names", None)
            
            # Process body_names: ensure it's a list of strings
            if body_names is None or (isinstance(body_names, list) and len(body_names) == 0):
                # Default key body names for G1 (must match training config)
                body_names = [
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                    "left_wrist_yaw_link",
                    "right_wrist_yaw_link",
                    "left_shoulder_roll_link",
                    "right_shoulder_roll_link"
                ]
            elif isinstance(body_names, (list, tuple)):
                # Filter out None values and ensure all are strings
                body_names = [str(x) for x in body_names if x is not None and str(x) != "None"]
                if len(body_names) == 0:
                    # Fallback to default if all were None
                    body_names = [
                        "left_ankle_roll_link",
                        "right_ankle_roll_link",
                        "left_wrist_yaw_link",
                        "right_wrist_yaw_link",
                        "left_shoulder_roll_link",
                        "right_shoulder_roll_link"
                    ]
            else:
                # Convert to list if it's a single value
                body_names = [str(body_names)] if body_names is not None else []
            
            # Get name, default to "robot" if None
            name = "robot"
            if isinstance(params_dict, dict) and "asset_cfg" in params_dict:
                asset_cfg_dict = params_dict["asset_cfg"]
                if isinstance(asset_cfg_dict, dict):
                    name = asset_cfg_dict.get("name", "robot")
            if name is None or str(name) == "None":
                name = "robot"
            
            # Set clean asset_cfg
            if not isinstance(params_dict, dict):
                params_dict = {}
            params_dict["asset_cfg"] = {
                "name": str(name),
                "body_names": body_names
            }
            term_cfg["params"] = params_dict
        
        # Remove clip if it's None (C++ handles missing clip gracefully)
        if "clip" in term_cfg:
            if term_cfg["clip"] is None:
                del term_cfg["clip"]
            elif isinstance(term_cfg["clip"], list) and len(term_cfg["clip"]) == 0:
                del term_cfg["clip"]
            elif isinstance(term_cfg["clip"], np.ndarray) and term_cfg["clip"].size == 0:
                del term_cfg["clip"]
        
        # Ensure history_length is ALWAYS set and is an integer (CRITICAL for deploy)
        # This must be set for ALL observation terms
        if "history_length" not in term_cfg:
            term_cfg["history_length"] = 5 if has_amp_terms else 1
        elif term_cfg.get("history_length") is None:
            term_cfg["history_length"] = 5 if has_amp_terms else 1
        else:
            # Convert to int, ensuring it's never None
            try:
                term_cfg["history_length"] = int(term_cfg["history_length"])
            except (ValueError, TypeError):
                term_cfg["history_length"] = 5 if has_amp_terms else 1
        
        # Final validation - history_length MUST be an integer
        assert isinstance(term_cfg["history_length"], int), f"history_length must be int, got {type(term_cfg['history_length'])}"
        assert term_cfg["history_length"] > 0, f"history_length must be > 0, got {term_cfg['history_length']}"
        
        # Ensure scale is not empty
        if "scale" in term_cfg:
            if term_cfg["scale"] is None or (isinstance(term_cfg["scale"], list) and len(term_cfg["scale"]) == 0):
                # Recompute scale from observation dimensions
                obs_dims = obs_cfg.dims if hasattr(obs_cfg, 'dims') else (1, term_cfg.get("history_length", 1))
                term_cfg["scale"] = [1.0] * obs_dims[1]
        
        # Use deploy observation name (not training name)
        cfg["observations"][deploy_obs_name] = term_cfg
        if deploy_obs_name is not None and deploy_obs_name != "None":
            obs_order.append(deploy_obs_name)
    
    # Set use_gym_history and obs_order for AMP policies
    # Note: ObservationManager receives cfg["observations"], so these must be inside observations dict
    if has_amp_terms:
        # Set use_gym_history to false (concatenated history, not interleaved)
        # This ensures history is concatenated per term: [term1_h0, term1_h1, ..., term1_h4, term2_h0, ...]
        # Instead of interleaved: [term1_h0, term2_h0, ..., term1_h1, term2_h1, ...]
        cfg["observations"]["use_gym_history"] = False
        # Store observation order explicitly (critical for AMP policies) - filter out None values and ensure strings
        filtered_order = [str(x) for x in obs_order if x is not None and str(x) != "None"]
        if not filtered_order or len(filtered_order) == 0:
            # If obs_order is empty, this is a critical error - raise exception
            raise ValueError(
                f"CRITICAL: obs_order is empty! This means no observations were registered.\n"
                f"Available observations in env: {list(env.observation_manager.active_terms.get('policy', {}).keys())}\n"
                f"Registered observations: {registered_observations}\n"
                f"obs_order collected: {obs_order}\n"
                f"Check that observation terms are properly registered and match the expected names."
            )
        cfg["observations"]["obs_order"] = filtered_order

    # --- save config file ---
    filename = os.path.join(log_dir, "params", "deploy.yaml")
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    if not isinstance(cfg, dict):
        cfg = class_to_dict(cfg)
    
    print("\njoint_ids_map result: "+ str(cfg["joint_ids_map"]))
    # Format all values to ensure proper YAML serialization
    # This converts tuples to lists, numpy types to Python types, etc.
    
    # Final pass: ensure no Python-specific types remain
    def clean_yaml_types(obj):
        """Recursively clean Python-specific types for YAML."""
        if isinstance(obj, dict):
            return {k: clean_yaml_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [clean_yaml_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        else:
            return obj
    
    cfg = clean_yaml_types(cfg)
    
    # Final cleanup: remove None values and ensure proper types
    # CRITICAL: Do NOT modify critical top-level fields like joint_ids_map, obs_order, etc.
    # These fields are already correctly set above and should not be touched by cleanup
    
    def clean_cfg_recursive(obj):
        """Recursively clean config, removing None values from lists and ensuring proper types.
        NOTE: This function should NOT modify critical top-level fields like joint_ids_map.
        """
        if isinstance(obj, dict):
            cleaned = {}
            for k, v in obj.items():
                # Skip cleanup for critical top-level fields - they're already correct
                if k in ["joint_ids_map", "stiffness", "damping", "default_joint_pos", "step_dt"]:
                    cleaned[k] = v  # Keep as-is, no cleanup
                elif k == "observations" and isinstance(v, dict):
                    # Special handling for observations - preserve obs_order, use_gym_history, and history_length
                    cleaned_obs = {}
                    for obs_k, obs_v in v.items():
                        if obs_k in ["obs_order", "use_gym_history"]:
                            cleaned_obs[obs_k] = obs_v  # Keep as-is
                        elif isinstance(obs_v, dict):
                            # Clean observation term config, but preserve history_length
                            cleaned_term = {}
                            for term_k, term_v in obs_v.items():
                                if term_k == "history_length":
                                    cleaned_term[term_k] = term_v  # Keep as-is (already validated)
                                elif term_k == "params" and isinstance(term_v, dict):
                                    # Clean params but preserve command_name
                                    cleaned_params = {}
                                    for param_k, param_v in term_v.items():
                                        if param_k == "command_name":
                                            cleaned_params[param_k] = param_v  # Keep as-is
                                        else:
                                            cleaned_params[param_k] = clean_cfg_recursive(param_v)
                                    cleaned_term[term_k] = cleaned_params
                                else:
                                    cleaned_term[term_k] = clean_cfg_recursive(term_v)
                            cleaned_obs[obs_k] = cleaned_term
                        else:
                            cleaned_obs[obs_k] = clean_cfg_recursive(obs_v)
                    cleaned[k] = cleaned_obs
                else:
                    cleaned_v = clean_cfg_recursive(v)
                    # Always keep critical fields, even if they're None (for joint_ids)
                    if cleaned_v is not None or k in ["joint_ids"]:  # joint_ids can be None
                        cleaned[k] = cleaned_v
            return cleaned
        elif isinstance(obj, (list, tuple)):
            # Filter None values but preserve list structure
            filtered = [clean_cfg_recursive(item) for item in obj if item is not None]
            return filtered
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    # Apply final cleanup - but preserve critical fields
    cfg = clean_cfg_recursive(cfg)

    # Post-cleanup validation: Ensure critical fields are not empty and have correct types
    if "joint_ids_map" in cfg:
        if not isinstance(cfg["joint_ids_map"], list):
            raise ValueError(f"CRITICAL: joint_ids_map is not a list! Got type: {type(cfg['joint_ids_map'])}")
        if len(cfg["joint_ids_map"]) == 0:
            raise ValueError(
                f"CRITICAL: joint_ids_map is empty after cleanup! This will cause deployment to fail.\n"
                f"Original joint_ids_map had {len(joint_ids_map)} elements.\n"
                f"Check that cleanup function is not incorrectly filtering values."
            )
        # Validate all elements are integers
        for i, val in enumerate(cfg["joint_ids_map"]):
            if not isinstance(val, (int, np.integer)):
                raise ValueError(f"CRITICAL: joint_ids_map[{i}] is not an integer! Got type: {type(val)}, value: {val}")
    
    if "observations" in cfg:
        if "obs_order" in cfg["observations"]:
            if not isinstance(cfg["observations"]["obs_order"], list):
                raise ValueError(f"CRITICAL: obs_order is not a list! Got type: {type(cfg['observations']['obs_order'])}")
            if len(cfg["observations"]["obs_order"]) == 0:
                raise ValueError("CRITICAL: obs_order is empty after cleanup! This will cause deployment to fail.")
        
        # Validate all observation terms have history_length
        for obs_name, obs_cfg in cfg["observations"].items():
            if obs_name in ["obs_order", "use_gym_history"]:
                continue  # Skip metadata fields
            if isinstance(obs_cfg, dict):
                if "history_length" not in obs_cfg:
                    raise ValueError(f"CRITICAL: Observation '{obs_name}' missing history_length!")
                if not isinstance(obs_cfg["history_length"], int):
                    raise ValueError(f"CRITICAL: Observation '{obs_name}' history_length is not int! Got: {type(obs_cfg['history_length'])}")
                if obs_cfg["history_length"] <= 0:
                    raise ValueError(f"CRITICAL: Observation '{obs_name}' history_length <= 0! Got: {obs_cfg['history_length']}")
                
                # Validate velocity_commands has command_name
                if obs_name in ["velocity_commands", "keyboard_velocity_commands"]:
                    if "params" not in obs_cfg or not isinstance(obs_cfg["params"], dict):
                        raise ValueError(f"CRITICAL: Observation '{obs_name}' missing params dict!")
                    if "command_name" not in obs_cfg["params"]:
                        raise ValueError(f"CRITICAL: Observation '{obs_name}' missing command_name in params!")
                    if obs_cfg["params"]["command_name"] != "base_velocity":
                        raise ValueError(f"CRITICAL: Observation '{obs_name}' command_name is not 'base_velocity'! Got: {obs_cfg['params']['command_name']}")
    
    # Ensure critical fields are correct
    if "observations" in cfg:
        # Ensure obs_order has no None values
        if "obs_order" in cfg["observations"]:
            cfg["observations"]["obs_order"] = [str(x) for x in cfg["observations"]["obs_order"] if x is not None and str(x) != "None"]
        
        # Ensure use_gym_history is boolean, not None
        if "use_gym_history" in cfg["observations"] and cfg["observations"]["use_gym_history"] is None:
            cfg["observations"]["use_gym_history"] = False
        
        # Remove clip: null from all observation terms
        for obs_name, obs_cfg in cfg["observations"].items():
            if isinstance(obs_cfg, dict) and "clip" in obs_cfg and obs_cfg["clip"] is None:
                del obs_cfg["clip"]
    
    # Use SafeDumper to avoid Python-specific tags like !!python/tuple
    # and ensure clean YAML output
    with open(filename, "w") as f:
        yaml.dump(cfg, f, default_flow_style=None, sort_keys=False, allow_unicode=True, Dumper=yaml.SafeDumper)

