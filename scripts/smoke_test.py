"""Exercise skill storage, scoring, and the agent loop without APIs or datasets."""
from pathlib import Path
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'eval'))
from medmem_agent.agent import AgentLoop
from medmem_agent.llm import LLMResponse
from medmem_agent.memory import ConversationMemory
from medmem_agent.tools.registry import ToolRegistry
from medmem_agent.evolution.config import load_evo_config
from medmem_agent.evolution.storage import SkillStore
from eval_utils import mc_exact_match

class FakeLLM:
    def generate(self, **kwargs):
        assert "{tool_list}" not in kwargs["system_prompt"]
        assert "describe_skill" in kwargs["system_prompt"]
        return LLMResponse(content='<reasoning>The synthetic answer is A.\nNEXT TAG: <response></reasoning><response>A</response>')

def main():
    root=Path(__file__).resolve().parents[1]
    config=load_evo_config(str(root/'configs/evolution/default.json'),str(root/'configs/evolution/offline.json'))
    assert (config.buffer.window_size,config.buffer.capacity)==(30,20)
    with tempfile.TemporaryDirectory(prefix='skemex-smoke-') as tmp:
        config.storage.root_dir='cache'
        config.storage.experiment_root='memory'
        store=SkillStore(repo_root=tmp,evo_config=config)
        store.ensure_initialized()
        assert store.categories_path.exists()
        agent=AgentLoop(llm=FakeLLM(),tool_registry=ToolRegistry(dataset_name='synthetic',tools={}),memory=ConversationMemory(),max_steps=7,output_dir=tmp)
        answer=agent.run('Synthetic smoke check: respond with option A.')
        assert mc_exact_match(answer,'A'),repr(answer)
        assert not mc_exact_match('B','A')
    print('PASS: paper configuration, packaged seeds, agent loop, and label scoring (no API calls).')

if __name__=='__main__': main()
