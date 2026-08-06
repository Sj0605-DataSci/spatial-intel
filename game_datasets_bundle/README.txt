Game world-model datasets.

Per game folder:
  base_image.png     initial-state frame
  themed_ground.png  fixed background layer (identical across all frames)
  assets/sprites/    individual object sprites (transparent RGBA) used to compose frames
  assets/background.png, assets/_bg_reference.png  raw + reference backgrounds
  states/            every state rendered as a frame (deterministic)
  transitions.jsonl  one row per (obs_t --action--> obs_next):
     from,to = state ids; action/action_name/target;
     control = player key (W/A/S/D move, Space = act on object under player);
     reward; done; effects = {moved_player, hidden_object, completed_object,
                int_changes, bool_flips}; obs_t/obs_next = frame paths
  action_vocab.json  action id <-> (base action, target object, control key)

Controls (how a child plays): W/A/S/D = up/left/down/right,
Space = context-sensitive action on the object in the player's cell.
  states.jsonl       per-state metadata (position, vars, is_goal)
  meta.json          counts + engine report (mode: full|sampled)
  source/            original RDDL, YAML and prompt files for the game
