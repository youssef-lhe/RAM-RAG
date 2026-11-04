khlifif 3id milad said 

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
!!! ser l mistral.ai khod api key , o hta f .env 
streamlit run RAMchat.py
docker compose up --build
http://localhost:8501
