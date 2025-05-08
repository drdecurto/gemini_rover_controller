#!/usr/bin/env python3
# gemini_rover_controller.py - Intelligent rover control using Gemini
# Based on Earth Rover SDK: https://github.com/frodobots-org/earth-rovers-sdk

import os
import time
import json
import base64
import requests
import argparse
import logging
import math
from datetime import datetime
from PIL import Image, ImageGrab
from dotenv import load_dotenv
import google.generativeai as genai
from retry_utils import retry_with_backoff, RetryException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rover_mission.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("gemini_rover_controller")

# Configuration Constants
DEFAULT_SDK_BASE_URL = "http://localhost:8000"
DEFAULT_PLANNING_INTERVAL = 5  # seconds
DEFAULT_SCREENSHOT_DIR = "screenshots"
MAX_RETRIES = 3
RETRY_DELAY = 2.0

class GeminiRoverController:
    def __init__(self, sdk_base_url, gemini_api_key, planning_interval=DEFAULT_PLANNING_INTERVAL):
        """
        Initialize the Gemini-powered rover controller.
        
        Args:
            sdk_base_url: Base URL for the Earth Rover SDK API
            gemini_api_key: API key for Google's Gemini API
            planning_interval: Interval (in seconds) between replanning cycles
        """
        self.sdk_base_url = sdk_base_url
        self.planning_interval = planning_interval
        self.screenshot_dir = DEFAULT_SCREENSHOT_DIR
        
        # Ensure screenshot directory exists
        os.makedirs(self.screenshot_dir, exist_ok=True)
        
        # Initialize Gemini
        try:
            genai.configure(api_key=gemini_api_key)
            self.gemini_model = genai.GenerativeModel(
                model_name="gemini-2.5-pro-exp-03-25",
                generation_config={
                    "temperature": 0.2,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 8192,
                }
            )
            logger.info("Gemini API configured successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            raise
        
        # Initialize mission state
        self.mission_active = False
        self.checkpoints = []
        self.current_checkpoint = 0
        self.last_telemetry = None
        self.mission_completed = False
        
    def start_mission(self):
        """Start the mission and retrieve checkpoint information."""
        try:
            # Start the mission
            response = requests.post(f"{self.sdk_base_url}/start-mission")
            if response.status_code != 200:
                logger.error(f"Failed to start mission: {response.text}")
                return False
            
            logger.info("Mission started successfully")
            self.mission_active = True
            
            # Get checkpoints list
            checkpoints_response = requests.post(f"{self.sdk_base_url}/checkpoints-list")
            if checkpoints_response.status_code != 200:
                logger.error(f"Failed to get checkpoints: {checkpoints_response.text}")
                return False
            
            checkpoints_data = checkpoints_response.json()
            self.checkpoints = checkpoints_data.get("checkpoints_list", [])
            self.current_checkpoint = checkpoints_data.get("latest_scanned_checkpoint", 0)
            
            logger.info(f"Retrieved {len(self.checkpoints)} checkpoints")
            logger.info(f"Current checkpoint: {self.current_checkpoint}")
            
            return True
        except Exception as e:
            logger.error(f"Error starting mission: {e}")
            return False
    
    def get_telemetry(self):
        """Get current telemetry data from the rover."""
        try:
            response = requests.get(f"{self.sdk_base_url}/data")
            if response.status_code != 200:
                logger.error(f"Failed to get telemetry: {response.text}")
                return None
            
            telemetry = response.json()
            self.last_telemetry = telemetry
            return telemetry
        except Exception as e:
            logger.error(f"Error getting telemetry: {e}")
            return None
    
    def take_screenshot(self):
        """Capture a screenshot of the interface."""
        try:
            # Create timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.screenshot_dir}/rover_view_{timestamp}.png"
            
            # Capture the screen
            screenshot = ImageGrab.grab()
            screenshot.save(filename)
            logger.info(f"Screenshot saved to {filename}")
            
            return filename
        except Exception as e:
            logger.error(f"Error taking screenshot: {e}")
            return None
    
    def encode_image(self, image_path):
        """Encode an image to base64 for the Gemini API."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image: {e}")
            return None
    
    def send_command(self, linear, angular):
        """
        Send a movement command to the rover.
        
        Args:
            linear: Linear velocity (-1 to 1)
            angular: Angular velocity (-1 to 1)
        
        Returns:
            True if command was sent successfully, False otherwise
        """
        try:
            command_data = {
                "command": {
                    "linear": linear,
                    "angular": angular
                }
            }
            
            response = requests.post(
                f"{self.sdk_base_url}/control",
                headers={"Content-Type": "application/json"},
                data=json.dumps(command_data)
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to send command: {response.text}")
                return False
            
            logger.info(f"Command sent: linear={linear}, angular={angular}")
            return True
        except Exception as e:
            logger.error(f"Error sending command: {e}")
            return False
    
    def get_next_checkpoint(self):
        """Get the next checkpoint to navigate to."""
        if not self.checkpoints or self.current_checkpoint >= len(self.checkpoints):
            return None
        
        next_checkpoint_idx = self.current_checkpoint
        if next_checkpoint_idx < len(self.checkpoints):
            return self.checkpoints[next_checkpoint_idx]
        return None
    
    def calculate_distance_bearing(self, lat1, lon1, lat2, lon2):
        """
        Calculate distance and bearing between two GPS coordinates.
        
        Args:
            lat1, lon1: Source position (current position)
            lat2, lon2: Destination position (next checkpoint)
        
        Returns:
            (distance_in_meters, bearing_in_degrees)
        """
        # Convert to radians
        lat1, lon1 = math.radians(float(lat1)), math.radians(float(lon1))
        lat2, lon2 = math.radians(float(lat2)), math.radians(float(lon2))
        
        # Earth radius in meters
        R = 6371000
        
        # Calculate distance using Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = R * c  # Distance in meters
        
        # Calculate bearing
        y = math.sin(dlon) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        bearing = math.atan2(y, x)
        
        # Convert to degrees
        bearing = math.degrees(bearing)
        bearing = (bearing + 360) % 360  # Normalize to 0-360
        
        return distance, bearing
    
    @retry_with_backoff(max_retries=3, initial_delay=5.0, backoff_factor=2.0)
    def plan_next_moves(self, telemetry, screenshot_path):
        """
        Use Gemini to plan the next moves based on telemetry and visual data.
        
        Args:
            telemetry: Rover telemetry data
            screenshot_path: Path to the current screenshot
        
        Returns:
            A list of movement commands to execute
        """
        try:
            # First, get next checkpoint information
            next_checkpoint = self.get_next_checkpoint()
            if not next_checkpoint:
                logger.info("No more checkpoints to navigate to")
                return []
            
            # Calculate distance and bearing to next checkpoint
            current_lat = telemetry.get("latitude")
            current_lon = telemetry.get("longitude")
            target_lat = float(next_checkpoint.get("latitude"))
            target_lon = float(next_checkpoint.get("longitude"))
            
            distance, bearing = self.calculate_distance_bearing(
                current_lat, current_lon, target_lat, target_lon
            )
            
            # Current rover orientation (in degrees)
            current_orientation = telemetry.get("orientation")
            
            # Encode screenshot for Gemini
            base64_image = self.encode_image(screenshot_path)
            if not base64_image:
                logger.error("Failed to encode screenshot for Gemini")
                return []
            
            # Prepare prompt for Gemini
            prompt = f"""
            You are the AI controller for an Earth Rover in the FrodoBots challenge.
            
            CURRENT TELEMETRY:
            - Battery: {telemetry.get('battery')}%
            - GPS Location: {current_lat}, {current_lon}
            - Orientation: {current_orientation}° (0° is North, 90° is East)
            - Speed: {telemetry.get('speed')}
            - Signal strength: {telemetry.get('signal_level')}
            
            NAVIGATION TASK:
            - Current checkpoint: {self.current_checkpoint + 1} of {len(self.checkpoints)}
            - Next checkpoint coordinates: {target_lat}, {target_lon}
            - Distance to checkpoint: {distance:.2f} meters
            - Bearing to checkpoint: {bearing:.2f}° (0° is North, 90° is East)
            
            MOVEMENT COMMANDS:
            The rover accepts commands with two parameters:
            - linear: controls forward/backward motion (-1 to 1)
            - angular: controls turning (-1 to 1, negative is left, positive is right)
            
            Looking at the telemetry data and the screenshot, plan the optimal movements to reach the next checkpoint.
            Return your answer in JSON format with 1-3 movement commands to execute in sequence:
            
            [
                {"linear": <value>, "angular": <value>, "duration": <seconds>},
                ...
            ]
            
            Keep your reasoning short and focus on providing practical movement commands.
            """
            
            # Send to Gemini
            response = self.gemini_model.generate_content(
                [prompt, {"mime_type": "image/jpeg", "data": base64_image}]
            )
            
            # Extract movement commands from response
            response_text = response.text
            
            # Find JSON in response
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                try:
                    commands = json.loads(json_str)
                    logger.info(f"Successfully planned {len(commands)} movement commands")
                    return commands
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse Gemini response as JSON: {e}")
            
            logger.error("No valid movement commands found in Gemini response")
            # Fallback to simple navigation
            return self.fallback_navigation(current_orientation, bearing, distance)
            
        except Exception as e:
            logger.error(f"Error in planning: {e}")
            return []
    
    def fallback_navigation(self, current_orientation, target_bearing, distance):
        """
        Generate fallback navigation commands if Gemini planning fails.
        
        Args:
            current_orientation: Current rover orientation in degrees
            target_bearing: Target bearing in degrees
            distance: Distance to target in meters
        
        Returns:
            List of movement commands
        """
        # Calculate angle difference (how much we need to turn)
        angle_diff = (target_bearing - current_orientation + 360) % 360
        
        # Convert to -180 to 180 range
        if angle_diff > 180:
            angle_diff -= 360
        
        commands = []
        
        # First, turn to face the target
        if abs(angle_diff) > 10:
            # Determine turn direction and duration
            angular = 0.5 if angle_diff > 0 else -0.5
            turn_duration = min(5, abs(angle_diff) / 30)  # Max 5 seconds
            
            commands.append({
                "linear": 0,
                "angular": angular,
                "duration": turn_duration
            })
        
        # Then move forward
        forward_duration = min(5, distance / 10)  # Max 5 seconds
        commands.append({
            "linear": 0.5,
            "angular": 0,
            "duration": forward_duration
        })
        
        return commands
    
    def execute_mission(self):
        """Execute the full mission using Gemini for planning."""
        if not self.start_mission():
            logger.error("Failed to start mission")
            return False
        
        logger.info("Starting mission execution")
        
        while not self.mission_completed:
            # Get current telemetry
            telemetry = self.get_telemetry()
            if not telemetry:
                logger.error("Failed to get telemetry, retrying...")
                time.sleep(2)
                continue
            
            # Take a screenshot
            screenshot_path = self.take_screenshot()
            if not screenshot_path:
                logger.error("Failed to take screenshot, retrying...")
                time.sleep(2)
                continue
            
            # Check mission completion
            current_checkpoint_status = requests.post(f"{self.sdk_base_url}/checkpoints-list").json()
            self.current_checkpoint = current_checkpoint_status.get("latest_scanned_checkpoint", 0)
            
            if self.current_checkpoint >= len(self.checkpoints):
                logger.info("All checkpoints reached! Mission completed successfully.")
                self.mission_completed = True
                break
            
            # Plan next moves using Gemini
            try:
                movement_commands = self.plan_next_moves(telemetry, screenshot_path)
                
                # Execute each command in sequence
                for cmd in movement_commands:
                    linear = cmd.get("linear", 0)
                    angular = cmd.get("angular", 0)
                    duration = cmd.get("duration", 1)
                    
                    logger.info(f"Executing command: linear={linear}, angular={angular}, duration={duration}s")
                    self.send_command(linear, angular)
                    
                    # Wait for the specified duration
                    time.sleep(duration)
                    
                    # Stop the rover after executing the command
                    self.send_command(0, 0)
            except RetryException as e:
                logger.warning(f"Planning failed after retries: {e}")
                # Use fallback navigation
                logger.info("Using fallback navigation")
                next_checkpoint = self.get_next_checkpoint()
                if next_checkpoint:
                    current_lat = telemetry.get("latitude")
                    current_lon = telemetry.get("longitude")
                    target_lat = float(next_checkpoint.get("latitude"))
                    target_lon = float(next_checkpoint.get("longitude"))
                    distance, bearing = self.calculate_distance_bearing(
                        current_lat, current_lon, target_lat, target_lon
                    )
                    fallback_commands = self.fallback_navigation(
                        telemetry.get("orientation"), bearing, distance
                    )
                    
                    # Execute fallback commands
                    for cmd in fallback_commands:
                        self.send_command(cmd.get("linear", 0), cmd.get("angular", 0))
                        time.sleep(cmd.get("duration", 1))
                        self.send_command(0, 0)
            
            # Wait before next planning cycle
            logger.info(f"Waiting {self.planning_interval}s before next planning cycle")
            time.sleep(self.planning_interval)
        
        return True

def main():
    """Main entry point for the Gemini-powered rover controller."""
    parser = argparse.ArgumentParser(description="Gemini-powered Earth Rover controller")
    parser.add_argument("--sdk-url", default=DEFAULT_SDK_BASE_URL, help="Earth Rover SDK base URL")
    parser.add_argument("--interval", type=float, default=DEFAULT_PLANNING_INTERVAL, 
                        help="Planning interval in seconds")
    parser.add_argument("--gemini-key", help="Gemini API key (or set GEMINI_API_KEY env var)")
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Get Gemini API key from args or environment
    gemini_api_key = args.gemini_key or os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        logger.error("Gemini API key not provided. Use --gemini-key or set GEMINI_API_KEY env var")
        return 1
    
    # Create and run the controller
    try:
        controller = GeminiRoverController(
            sdk_base_url=args.sdk_url,
            gemini_api_key=gemini_api_key,
            planning_interval=args.interval
        )
        
        success = controller.execute_mission()
        if success:
            logger.info("Mission completed successfully")
            return 0
        else:
            logger.error("Mission failed")
            return 1
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        return 1

if __name__ == "__main__":
    exit(main())