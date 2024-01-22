conda init -y
source ~/.bashrc
conda create -y -n aces python=3.10
echo "conda activate aces" > ~/.bashrc
source ~/.bashrc
git clone https://github.com/Julien-pour/OpenELM.git
cd OpenELM
git checkout imgep-qdaif
conda install pytorch torchvision torchaudio cpuonly -c pytorch
pip install -r requirements.txt
pip install -e .
echo 'export OPENAI_API_KEY="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"' >> ~/.bashrc
source ~/.bashrc
# download models
python -c "from transformers import LlamaTokenizer, LlamaForCausalLM; model_path = 'openlm-research/open_llama_3b_v2'; tokenizer = LlamaTokenizer.from_pretrained(model_path); import torch; model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map='auto')"
echo "export PATH=$PATH:/sbin" >> ~/.bashrc
source ~/.bashrc
