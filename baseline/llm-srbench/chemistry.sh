export OPENAI_API_KEY="${OPENAI_API_KEY:?Please set OPENAI_API_KEY}"
cd /home/xaa5sgh/symregression/baseline/llm-srbench

# Bio-Pop-Growth Dataset
# python eval.py --dataset bio_pop_growth --searcher_config configs/llmdirect_llama31_8b.yaml --local_llm_port 10005
# python eval.py --dataset bio_pop_growth --searcher_config configs/lasr_llama31_8b.yaml --local_llm_port 10005
# python eval.py --dataset bio_pop_growth --searcher_config configs/sga_llama31_8b.yaml --local_llm_port 10005
# python eval.py --dataset bio_pop_growth --searcher_config configs/llmsr_opus-4-6.yaml --dataset_snapshot_dir /home/xaa5sgh/symregression/datasets/llm-srbench > bio_pop_growth_llmsr_opus-4-6.log 2>&1

# # Chem React Kinetics
# python eval.py --dataset chem_react --searcher_config configs/llmdirect_llama31_8b.yaml --local_llm_port 10005
# python eval.py --dataset chem_react --searcher_config configs/lasr_llama31_8b.yaml --local_llm_port 10005
# python eval.py --dataset chem_react --searcher_config configs/sga_llama31_8b.yaml --local_llm_port 10005
# python eval.py --dataset chem_react --searcher_config configs/llmsr_llama31_8b.yaml --local_llm_port 10005
python eval.py --dataset chem_react --searcher_config configs/llmsr_opus-4-6.yaml --dataset_snapshot_dir /home/xaa5sgh/symregression/datasets/llm-srbench > chem_react_llmsr_opus-4-6.log 2>&1

# # Matsci SS
# python eval.py --dataset matsci --searcher_config configs/llmdirect_llama31_8b.yaml --local_llm_port 10005
# python eval.py --dataset matsci --searcher_config configs/lasr_llama31_8b.yaml --local_llm_port 10005
# python eval.py --dataset matsci --searcher_config configs/sga_llama31_8b.yaml --local_llm_port 10005
# python eval.py --dataset matsci --searcher_config configs/llmsr_llama31_8b.yaml --local_llm_port 10005

# # Phys oscillator
# python eval.py --dataset phys_osc --searcher_config configs/llmdirect_llama31_8b.yaml --local_llm_port 10005
# python eval.py --dataset phys_osc --searcher_config configs/lasr_llama31_8b.yaml --local_llm_port 10005
# python eval.py --dataset phys_osc --searcher_config configs/sga_llama31_8b.yaml --local_llm_port 10005
# python eval.py --dataset phys_osc --searcher_config configs/llmsr_llama31_8b.yaml --local_llm_port 10005