# EmoFlow
## Mood-Based Music Generation with Facial and Speech Emotion Recognition

[streamlit-app-demo](https://github.com/user-attachments/assets/a3979c58-313e-44fc-900a-96300b2aa566)

## ⭐  Quick Start  ⭐

If you want to use the app on google colab, you should clone the repo:
```bash
!git clone https://github.com/muratbakirr/EmoFlow.git
```
Set up a secure tunnel to expose a local web server to the internet using *cloudflared* tool:
```bash
!wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
!chmod +x cloudflared-linux-amd64
!nohup /content/cloudflared-linux-amd64 tunnel --url http://localhost:8501 &
```
Install the dependicies from requirements.txt file:
```bash
%cd '/content/EmoFlow'
!pip install -r ./requirements.txt
```
Run the app.py and get the url to enter into the app:
```bash
!streamlit run app.py &>/content/logs.txt &
!grep -o 'https://.*\.trycloudflare.com' /content/nohup.out | head -n 1 | xargs -I {} echo "Your tunnel url {}"
```


-----------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------

If you plan to use the app regularly, you should clone the repo and run it locally:
```bash
!git clone https://github.com/muratbakirr/EmoFlow.git
!pip install -r ./requirements.txt
!streamlit run app.py
```
