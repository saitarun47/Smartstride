# Smartstride

**Smart Intelligence** is an AI-powered system (agentic ai) that leverages Strava activity data to deliver personalized, and data-driven feedback for athletes. Designed for runners, cyclists, and swimmers, this project combines state-of-the-art retrieval augmented generation (RAG), vector search, and automated email reporting to help users unlock their full athletic potential.

**Final output as a mail to the use**
<img width="1626" height="833" alt="image" src="https://github.com/user-attachments/assets/d5608da2-0255-4eea-ad7e-8dabde1d0d58" />


## Features

- **Strava Data Integration:** Seamlessly fetches and preprocesses your Strava activities for deep analysis.
- **RAG-Powered Insights:** Uses vector databases and transformer models to answer natural language queries about your training history.
- **Personalized Coaching:** Provides actionable recommendations, goal setting, and motivational feedback tailored to your fitness level and recent trends.
- **Automated Email Reports:** Generates stunning, blog-style plain text performance reports and sends them directly to your inbox.
- **Multi-Sport Support:** Specialized analysis for running, cycling, and swimming, including discipline-specific metrics and cross-training advice.
- **Agentic Memory:** Remembers your sessions and adapts feedback over time for continuous improvement.

## 🏗️ Tech Stack

- **Python 3.12+**
- [Agno AI](https://github.com/agnolabs/agno) agent framework
- [Stravalib](https://github.com/hozn/stravalib) for Strava API integration
- [Sentence Transformers](https://www.sbert.net/) for semantic search
- [ChromaDB](https://www.trychroma.com/) & [LanceDB](https://lancedb.com/) for vector storage
- [Google Gemini](https://deepmind.google/technologies/gemini/) for advanced LLM reasoning
- [Pandas](https://pandas.pydata.org/) for data wrangling


## 📊 Example Output

- **Comprehensive Run Analysis Email:**  
  - Latest performance breakdown with pace conversions  
  - Multi-run comparison with percentage improvements  
  - Clear metrics highlighting strengths and areas to improve  
  - Actionable, sport-specific recommendations  
  - Motivational, goal-oriented messaging

## 🛠️ How It Works

1. **Connect your Strava account** use the credentials and fetch recent activities.
2. **Analyze your data** using RAG tools and transformer models.
3. **Receive detailed feedback** and improvement plans via email.
4. **Track your progress** with agentic memory and historical trend analysis.

## 📦 Installation

```sh
git clone https://github.com/yourusername/smart-intelligence.git
pip install -r [requirements.txt](http://_vscodecontentref_/0)
add credentials in .env
python data.py
python main.py

