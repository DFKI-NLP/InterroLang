FROM python:3 
ADD test /
ADD Topk /

# RUN pip install -r requirements.txt
RUN pip install pytest
RUN ls
RUN python -m pytest test_topk.py
