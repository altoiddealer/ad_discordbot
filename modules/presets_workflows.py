from modules.utils_shared import load_file, shared_path
from modules.utils_misc import deep_merge

from modules.logs import import_track, get_logger; import_track(__file__, fp=True); log = get_logger(__name__)  # noqa: E702
logging = log

class Presets():
    def __init__(self):
        self.presets:dict = {}
        self.init()

    def init(self):
        data = load_file(shared_path.presets)
        presets_list = data.get('presets', [])
        for preset in presets_list:
            if not isinstance(preset, dict):
                log.warning("Encountered a non-dict response handling preset. Skipping.")
                continue
            name = preset.get('name')
            if name:
                self.presets[name] = preset

    def is_simple_step_preset(self, preset: dict) -> bool:
        """True if preset only contains 'name' and 'steps'."""
        return (isinstance(preset, dict) and
                set(preset.keys()) <= {"name", "steps"} and
                isinstance(preset.get("steps"), list))

    def get_preset(self, preset_name: str, default=None) -> dict:
        return self.presets.get(preset_name, default or {})

    def apply_preset(self, config: dict) -> dict:
        preset_name = config.get("preset")
        if not preset_name:
            return config
        preset = self.get_preset(preset_name)
        if not preset:
            log.warning(f"Preset '{preset_name}' not found.")
            return config
        # Merge preset with overrides (config wins)
        merged = deep_merge(preset, config)
        merged.pop("preset", None)
        return merged

    def apply_presets(self, config):
        """
        Recursively applies presets to all dicts found in the config (which can be a dict or list).
        Returns a new config structure with all presets applied.
        """
        if isinstance(config, dict):
            # Apply preset and get merged result
            config = self.apply_preset(config)
            # Recurse into each value and update it
            for key, value in config.items():
                config[key] = self.apply_presets(value)
            return config
        elif isinstance(config, list):
            new_list = []
            for item in config:
                # list item is a dict containing only 'preset'
                if (isinstance(item, dict) and set(item.keys()) == {"preset"}):
                    preset_name = item["preset"]
                    preset = self.get_preset(preset_name)
                    # insert the preset list items
                    if self.is_simple_step_preset(preset):
                        preset_steps = preset.get("steps", [])
                        new_list.extend(self.apply_presets(pstep) for pstep in preset_steps)
                        continue  # Skip adding the original item
                # Normal recursive case
                new_list.append(self.apply_presets(item))
            return new_list
        else:
            # Base case: return scalar values as-is
            return config

bot_presets = Presets()


class Workflows():
    def __init__(self):
        self.workflows:dict = {}
        self._init()

    def _init(self):
        data = load_file(shared_path.workflows)
        workflows_list = data.get('workflows', [])
        for workflow in workflows_list:
            if not isinstance(workflow, dict):
                log.warning("Encountered a non-dict Workflow. Skipping.")
                continue
            name = workflow.get('name')
            if not name:
                log.warning("Encountered a Workflow without a 'name'. Skipping.")
                continue
            steps = workflow.get('steps')
            if not steps:
                log.warning("Encountered a Workflow without any 'steps'. Skipping.")
                continue
            self.workflows[name] = workflow
            self.workflows[name]['steps'] = bot_presets.apply_presets(steps)

    def get_workflow_steps_for(self, workflow_name:str) -> dict:
        workflow = self.workflows.get(workflow_name)
        if not workflow:
            raise RuntimeError(f'[Workflows] Workflow not found "{workflow_name}"')
        if not isinstance(workflow, dict):
            raise ValueError(f'[Workflows] Invalid format for workflow "{workflow_name}" (must be dict)')
        workflow_steps = workflow.get('steps')
        if not isinstance(workflow_steps, list):
            raise ValueError(f'[Workflows] Invalid structure for workflow "{workflow_name}" (include a "step" key)')
        return workflow_steps

bot_workflows = Workflows()
