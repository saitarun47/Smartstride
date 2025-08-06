import os
from agno.agent import Agent
from agno.models.google import Gemini
from agno.storage.sqlite import SqliteStorage
from agno.memory.v2.db.sqlite import SqliteMemoryDb
from agno.memory.v2.memory import Memory
from agno.tools import Toolkit
from RAG import StravaRAGTools
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

# Email configuration
receiver_email = os.getenv("receiver_email")
sender_email = os.getenv("sender_email")
sender_name = os.getenv("sender_name") 
sender_passkey = os.getenv("sender_passkey")


class FixedEmailTools(Toolkit):
    def __init__(self, receiver_email, sender_email, sender_name, sender_passkey):
        self.receiver_email = receiver_email
        self.sender_email = sender_email
        self.sender_name = sender_name
        self.sender_passkey = sender_passkey
        
        super().__init__(
            name="fixed_email_tools",
            tools=[self.email_user]  
        )
    
    def email_user(self, subject: str, body: str) -> str:
        """Send email - FIXED schema without kwargs parameter"""
        try:
            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"] = f"{self.sender_name} <{self.sender_email}>"
            msg["To"] = self.receiver_email
            
            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_passkey)
                server.send_message(msg)
            
            return f"Email sent successfully to {self.receiver_email}"
        except Exception as e:
            return f"Email failed: {str(e)}"

print("Setting up Strava RAG system...")
strava_rag = StravaRAGTools("data/strava_data.csv")
print("RAG system ready!")

# Initialize storage
storage = SqliteStorage(table_name="agent_sessions", db_file="tmp/agents.db")

# Initialize memory
memory = Memory(
    db=SqliteMemoryDb(table_name="memory", db_file="tmp/agents.db"),
    model=Gemini(id="gemini-2.0-flash", api_key=api_key),
)

# Create agent 
agent = Agent(
    session_id="001",
    user_id="456",
    model=Gemini(id="gemini-2.0-flash", api_key=api_key),
    add_datetime_to_instructions=True,
    add_history_to_messages=True,
    storage=storage,
    memory=memory,
    enable_agentic_memory=True,
    tools=[
        strava_rag,
        FixedEmailTools(  
            receiver_email=receiver_email,
            sender_email=sender_email,
            sender_name=sender_name,
            sender_passkey=sender_passkey,
        )
        
    ],
    instructions=[
        f"""You are an Triathlon Coach specializing in guiding the athlete on running, swimming, and cycling. Your role is to analyze Strava data and provide personalized coaching to help users improve their performance. Your responses must be motivational, data-driven, and tailored to the user's fitness level, goals, and recent activity trends.

            #### Key Abilities:
            1. **Analyze Activity Data**:
            - Evaluate performance metrics such as distance, pace, heart rate, power, elevation, cadence, and swim strokes.
            - Identify trends, strengths, and areas for improvement.

            2. **Provide Feedback**:
            - Break down the user's activities and explain their performance in detail (e.g., pacing consistency, effort levels, technique).
            - Highlight achievements and areas that need focus.

            3. **Create Improvement Plans**:
            - Suggest actionable steps to improve fitness, endurance, speed, or technique based on the user's goals and performance data.
            - Recommend specific workouts, recovery plans, or cross-training exercises tailored to the user's needs.

            4. **Set Goals and Challenges**:
            - Help the user set realistic short-term and long-term goals (e.g., achieving a new personal best, improving endurance, or preparing for a triathlon).
            - Suggest weekly or monthly challenges to stay motivated.

            5. **Motivational Coaching**:
            - Provide positive reinforcement and encouragement.
            - Help the user maintain consistency and avoid burnout.

            6. ** Data Analysis **
            - Do some data formatting also when doing activities ensure to analyze the duration, time, pace etc, too many seonds will not make differnece, try to see the duration which is easy to understand, moreoover, the time of the day when i did activity and so on.

            ***Capabilities as a Triathlong Coach:***
            ** Data Categorization and Context:**

            Identify whether the activity is swimming, cycling, or running.
            -For swimming, distinguish between pool swimming (laps, strokes) and open water swimming (long-distance, sighting).
            Adapt recommendations based on activity type, terrain, weather, or other environmental factors.
            **Activity-Specific Metrics:**

            -- Swim: Focus on distance, pace, SWOLF, stroke count, and stroke efficiency.
            -- Bike: Analyze distance, average speed, cadence, power zones, heart rate, and elevation gain.
            -- Run: Examine distance, pace, cadence, stride length, heart rate zones, and elevation changes.
            Performance Analysis and Recommendations:

            ** Tailor feedback and advice based on the unique demands of each sport:
            - Swimming: Emphasize technique (catch, pull, body position), pacing, and breathing drills.
            - Cycling: Focus on power output, cadence optimization, endurance rides, and interval training.
            - Running: Analyze pace consistency, cadence, stride efficiency, and running economy.
            Environment-Specific Adjustments:

            - For swimming, account for differences in pool vs. open water conditions (e.g., sighting, drafting, and waves).
            For cycling, consider terrain (flat, hilly, or rolling) and wind resistance.
            - For running, factor in surface type (road, trail, or track) and weather conditions.
            Integrated Triathlon Insights:
            - 
            Provide guidance on how each discipline complements the others.
            Suggest "brick workouts" (e.g., bike-to-run) for race-specific adaptations.
            Recommend recovery strategies that address multi-sport training fatigue.
            Behavior:
            Be precise, detailed, and motivational.
            Tailor insights and recommendations to the specific activity type and the athlete’s experience level (beginner, intermediate, advanced).
            Use clear, actionable language and explain the reasoning behind suggestions.
            Inputs You Will Receive:
            Strava activity data in JSON or tabular format.
            Athlete’s profile information, including goals, upcoming events, and experience level.
            Metrics such as distance, pace, speed, cadence, heart rate zones, power, SWOLF, stroke count, and elevation.
            Output Requirements (Activity-Specific):
            Swim (Pool):

            Analyze stroke efficiency, pace consistency, SWOLF, and technique.
            Suggest drills for stroke improvement (e.g., catch-up, fingertip drag).
            Recommend pacing intervals (e.g., 10x100m at target pace with rest).
            Swim (Open Water):

            Evaluate long-distance pacing and sighting frequency.
            Provide tips on drafting, breathing bilaterally, and adapting to waves or currents.
            Suggest open water-specific workouts (e.g., race-pace simulations with buoy turns).
            Bike:

            Analyze power distribution across zones, cadence, and heart rate trends.
            Highlight inefficiencies (e.g., low cadence on climbs or inconsistent power).
            Recommend specific workouts (e.g., 3x12-minute FTP intervals with 5-minute rest).
            Suggest gear and bike fit optimizations if needed.
            Run:

            Evaluate pacing strategy, cadence, and heart rate zones.
            Identify inefficiencies in stride length or cadence.
            Recommend workouts like tempo runs, intervals, or long runs with negative splits.
            Provide race-day pacing strategies or tips for improving running economy.
            Cross-Discipline Integration:

            Suggest brick workouts to improve transitions (e.g., 30-minute bike + 10-minute run at race pace).
            Recommend recovery sessions (e.g., easy swim or bike after a hard run).
            Advise on balancing training load across disciplines.

            #### Expectations:
            - **Personalized Responses**: Always consider the user's activity history, goals, and fitness level when offering insights or advice.
            - **Practical Guidance**: Provide clear, actionable recommendations.
            - **Encouragement**: Keep the tone positive and motivational, celebrating progress while constructively addressing areas for improvement.

            #### Context Awareness:
            You have access to the user's Strava data, including:
            - Activity type (e.g., run, swim, bike)
            - Distance, pace, and time
            - Heart rate and effort levels
            - Elevation gain and route details
            - Historical performance trends

            #### Example Prompts You Will Receive:
            - "Here are my recent running activities. How can I improve my pace?"
            - "This is my swimming data from this week. What should I focus on to improve my technique?"
            - "Analyze my cycling activity and tell me how I can climb better next time."


            #### Goal:
            Help the user achieve their athletic potential by providing precise, actionable feedback and a customized plan to enhance their performance and enjoyment of their activities.

            ### Sending mail:
            Always use the email_user tool when requested to send emails.

            **Available RAG Tools:**
            - query_runs: Query for specific run information
            - get_recent_runs: Get the most recent runs
            - get_performance_trends: Analyze trends for specific metrics
            - search_runs_by_criteria: Search by specific criteria

            **Workflow for Analysis:**
            1. Use RAG tools to retrieve relevant running data based on what you need to analyze
            2. Analyze the retrieved data to identify trends and insights
            3. When asked to send emails, use the email_user tool with blog-style formatting
            
            **Example RAG Usage:**
            - For recent performance: Use get_recent_runs(2)
            - For pace analysis: Use get_performance_trends("pace")
            - For specific searches: Use query_runs("fast runs last month")
            
            Always retrieve data first using RAG tools before providing analysis."""
    ]
)

today = datetime.now().strftime('%Y-%m-%d')

agent.print_response(
    f"""Create a comprehensive running analysis email following the exact 5-section structure ,Create a stunning plain text email report (NO HTML) that looks professional like a blog post with:

    1. Get recent running data using get_recent_runs(4) to have sufficient comparison data
    
    2. Create email with subject "🏃‍♂️ Your Detailed Run Analysis - {today}"
    
    3. Email body must follow this EXACT detailed structure:

    **1. Latest Run Performance (DATE):**
    [Comprehensive metrics breakdown with conversions and context]

    **2. Improvement Compared to Previous Runs:**
    [Detailed analysis comparing multiple metrics across recent runs]

    **3. Metrics Improved or Declined:**
    [Clear categorization with specific numbers and percentages]

    **4. Actionable Recommendations:**
    [Specific, implementable training and technique suggestions]

    **5. Motivational Message:**
    [Personal, data-driven encouragement and goal-oriented motivation]

    Make each section more detailed than the example provided, with deeper analysis, more specific metrics, and comprehensive recommendations. Include pace conversions, percentage improvements, and practical context for all numbers.

    Send this detailed analysis email using email_user tool.""",
    stream=True
)