import os
import chromadb
from sentence_transformers import SentenceTransformer
from agno.tools import Toolkit
import pandas as pd
from typing import List

class StravaRAGTools(Toolkit):
    def __init__(self, strava_data_path: str = "data/strava_data.csv"):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = chromadb.Client()
        self.collection = None
        self.strava_data_path = strava_data_path
        
        # Initialize the vector database
        self._setup_vector_db()
        
        
        super().__init__(
            name="strava_rag_tools",
            tools=[
                self.get_chronological_recent_runs,  
                self.get_recent_runs,                
                self.query_runs,                     
                self.get_performance_trends,         
                self.search_runs_by_criteria        
                
            ]
        )
    
    def _setup_vector_db(self):
        """Initialize vector database with Strava data"""
        try:
            self.client.delete_collection("strava_runs")
        except:
            pass
        
        self.collection = self.client.create_collection("strava_runs")
        
        # Load and store Strava data
        strava_data = pd.read_csv(self.strava_data_path)
        
        for index, run in strava_data[strava_data['sport_type'] == 'Run'].iterrows():
            run_text = f"""
            Date: {run['start_date_local']}
            Distance: {run['distance']}m
            Duration: {run['moving_time']}s
            Average Speed: {run['average_speed']}m/s
            Max Speed: {run['max_speed']}m/s
            Elevation Gain: {run['total_elevation_gain']}m
            Type: {run['sport_type']}
            """
            
            self.collection.add(
                documents=[run_text],
                metadatas=[{
                    "date": str(run['start_date_local']),
                    "distance": float(run['distance']),
                    "type": "run",
                    "avg_speed": float(run['average_speed'])
                }],
                ids=[f"run_{index}"]
            )
    
    def get_chronological_recent_runs(self, n_runs: int = 1) -> str:
        """
        FIXED: Get runs by chronological order (Windows compatible, no timeouts)
        
        Args:
            n_runs: Number of recent runs to retrieve (default: 1)
            
        Returns:
            Formatted string with recent run data
        """
        try:
           
            strava_data = pd.read_csv(self.strava_data_path)
            strava_data['start_date_local'] = pd.to_datetime(strava_data['start_date_local'])
            
            recent_runs = strava_data[strava_data['sport_type'] == 'Run'].sort_values('start_date_local', ascending=False).head(n_runs)
            
            
            result = f"Your {n_runs} most recent runs:\n\n"
            for i, (_, run) in enumerate(recent_runs.iterrows()):
                result += f"""Run {i+1}:
                    Date: {run['start_date_local'].strftime('%Y-%m-%d')}
                    Distance: {run['distance']}m
                    Duration: {run['moving_time']}s
                    Average Speed: {run['average_speed']:.1f}m/s
                    Max Speed: {run['max_speed']:.1f}m/s
                    Elevation Gain: {run['total_elevation_gain']}m

                    """
            return result
            
        except Exception as e:
            return f"Error retrieving recent runs: {str(e)}"
    
    def get_recent_runs(self, n_runs: int = 2) -> str:
        """
        Get the most recent runs by chronological date order.
        
        Args:
            n_runs: Number of recent runs to retrieve (default: 2)
            
        Returns:
            Formatted string with recent run data sorted by date
        """
        try:
            strava_data = pd.read_csv(self.strava_data_path)
            strava_data['start_date_local'] = pd.to_datetime(strava_data['start_date_local'])
            
            recent_runs = strava_data[strava_data['sport_type'] == 'Run'].sort_values('start_date_local', ascending=False).head(n_runs)
            
            result = f"Your {n_runs} most recent runs (by date):\n"
            for i, (_, run) in enumerate(recent_runs.iterrows()):
                result += f"\nRun {i+1}:\n"
                result += f"Date: {run['start_date_local'].strftime('%Y-%m-%d')}\n"
                result += f"Distance: {run['distance']}m\n"
                result += f"Duration: {run['moving_time']}s\n"
                result += f"Average Speed: {run['average_speed']:.1f}m/s\n"
                result += f"Max Speed: {run['max_speed']:.1f}m/s\n"
                result += f"Elevation Gain: {run['total_elevation_gain']}m\n"
            
            return result
            
        except Exception as e:
            return f"Error retrieving recent runs: {str(e)}"
    
    def query_runs(self, query: str, n_results: int = 3) -> str:
        """
        Query the Strava runs database for specific information.
        
        Args:
            query: Natural language query about runs
            n_results: Number of runs to retrieve (default: 3)
        
        Returns:
            Formatted string with relevant run data
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            if results['documents'][0]:
                runs_data = "\n".join([f"Run {i+1}:\n{doc}\n" for i, doc in enumerate(results['documents'][0])])
                return f"Found {len(results['documents'][0])} relevant runs:\n\n{runs_data}"
            else:
                return "No runs found matching your query."
                
        except Exception as e:
            return f"Error querying runs: {str(e)}"
    
    def get_performance_trends(self, focus: str = "pace") -> str:
        """
        Analyze performance trends for specific metrics.
        
        Args:
            focus: What to focus on - "pace", "distance", "speed", or "endurance"
            
        Returns:
            Relevant runs for trend analysis
        """
        query_map = {
            "pace": "pace speed tempo running faster slower",
            "distance": "long runs endurance mileage distance training", 
            "speed": "fast runs maximum speed sprint intervals",
            "endurance": "long distance endurance stamina cardiovascular"
        }
        
        query = query_map.get(focus, "performance improvement progress")
        return self.query_runs(query, 4)
    
    def search_runs_by_criteria(self, criteria: str) -> str:
        """
        Search runs by specific criteria.
        
        Args:
            criteria: Search criteria description
            
        Returns:
            Runs matching the criteria
        """
        return self.query_runs(criteria, 3)
