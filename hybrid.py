import json
import random
import wikipedia
import re
import os
from difflib import SequenceMatcher

# Configure Wikipedia
wikipedia.set_lang("en")
wikipedia.set_rate_limiting(True)  # ta ky api over use na ho

class HybridChatbot:
    def __init__(self, intents_file):
        if not os.path.exists(intents_file):
            raise FileNotFoundError(f"JSON file not found at: {os.path.abspath(intents_file)}")
        
        self.intents = self.load_intents(intents_file)
        self.threshold = 0.75  # zada threshold taky sai match ho
        self.blacklist = ["exit", "quit", "bye"]  # Commands to end chat
        
    def load_intents(self, file_path):
        """Load intents from JSON file with error handling"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            # Validate JSON 
            if 'intents' not in data:
                raise ValueError("JSON file must contain 'intents' array")
                
            return data['intents']
        except Exception as e:
            print(f"Error loading intents: {str(e)}")
            return []
    
    def preprocess_text(self, text):
        """Clean and normalize text for better matching"""
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        return text
    
    def get_best_intent(self, user_input):
        """Find the most relevant intent with improved matching"""
        processed_input = self.preprocess_text(user_input)
        best_match = None
        highest_similarity = 0
        
        for intent in self.intents:
            for pattern in intent['patterns']:
                pattern_processed = self.preprocess_text(pattern)
                similarity = self.similarity(processed_input, pattern_processed)
                
                # Only consider jo threshold sy milty hai
                if similarity > highest_similarity and similarity >= self.threshold:
                    highest_similarity = similarity
                    best_match = intent
        
        return best_match if highest_similarity >= self.threshold else None
    
    def similarity(self, a, b):
        """Calculate text similarity with improved algorithm"""
        return SequenceMatcher(None, a, b).ratio()
    
    def get_wikipedia_answer(self, query):
        """Get Wikipedia answer with better query handling"""
        try:
            # First try direct search
            search_results = wikipedia.search(query)
            if not search_results:
                return None
                
            # Get summary of most relevant result
            summary = wikipedia.summary(search_results[0], sentences=2)
            return f"According to Wikipedia: {summary}"
            
        except wikipedia.DisambiguationError as e:
            # Handle ambiguous queries
            options = e.options[:3]
            return f"Multiple options found. Did you mean: {', '.join(options)}?"
            
        except wikipedia.PageError:
            return None
            
        except Exception as e:
            print(f"Wikipedia error: {str(e)}")
            return None
    
    def respond(self, user_input):
        """Generate response with improved logic"""
        # Check for exit commands
        if user_input.lower() in self.blacklist:
            return "Goodbye! Have a nice day.", "System"
        
        # First try intent matching
        intent = self.get_best_intent(user_input)
        if intent:
            return random.choice(intent['responses']), "Chatbot Knowledge"
        
        # Then try Wikipedia
        wiki_response = self.get_wikipedia_answer(user_input)
        if wiki_response:
            return wiki_response, "Wikipedia"
        
        # Final response
        fallbacks = [
            "I couldn't find information about that.",
            "I'm not sure about that topic.",
            "That's beyond my current knowledge."
        ]
        return random.choice(fallbacks), "System"
    
    def chat(self):
        """Run interactive chat session"""
        print("\n🤖 Hybrid Chatbot: Hello! Ask me anything or type 'exit' to quit.\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in self.blacklist:
                    print("\nChatbot: Goodbye! Have a great day.")
                    break
                
                response, source = self.respond(user_input)
                print(f"\nChatbot: {response}")
                print(f"[Source: {source}]\n")
                
            except KeyboardInterrupt:
                print("\nChatbot: Session ended by user.")
                break
            except Exception as e:
                print(f"\nChatbot: Sorry, I encountered an error: {str(e)}")
                continue

if __name__ == "__main__":
    try:
        # Try multiple possible file locations
        possible_paths = [
            'intents_dataset_1000.json',
            os.path.join(os.path.dirname(__file__), 'intents_dataset_1000.json'),
            'E:/chatbot/chatbot/intents_dataset_1000.json'
        ]
        
        json_file = None
        for path in possible_paths:
            if os.path.exists(path):
                json_file = path
                break
                
        if not json_file:
            json_file = input("Please enter full path to intents JSON file: ")
        
        chatbot = HybridChatbot(json_file)
        chatbot.chat()
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Please ensure:")
        print("1. Your JSON file exists and is valid")
        print("2. You have internet connection for Wikipedia")
        print("3. Required packages are installed (wikipedia, python-Levenshtein)")