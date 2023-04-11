# Install polyjuice from local repo
git clone git@github.com:tongshuangwu/polyjuice.git
cd polyjuice
pip install -e .

# Download omw-1.4
python -m nltk.downloader omw-1.4

# Install spacy
pip install -U pip setuptools wheel
pip install -U spacy
python -m spacy download en_core_web_sm