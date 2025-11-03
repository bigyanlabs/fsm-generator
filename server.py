from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from google import genai
import os
import dotenv
dotenv.load_dotenv()

app = Flask(__name__)
CORS(app)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
client = genai.Client(api_key=GEMINI_API_KEY)
# models = client.models.list()
# print("List of available models:")

# for m in models:
#     print(m.name)

def generate(description):
    """Generate FSM from text description using Gemini with failsafe"""
    
    prompt = f"""You are an FSM (Finite State Machine) generator. Given a text description, generate a formal FSM specification.

Input: {description}

Output format (STRICT - follow exactly):
states: state1, state2, state3
transitions: state1 --event--> state2, state2 --event--> state3

Rules:
1. List ALL states separated by commas after "states:"
2. List ALL transitions after "transitions:"
3. Use format: from_state --event--> to_state
4. If no event label, use: from_state -> to_state
5. Use simple, single-word state names (lowercase with underscores)
6. Keep event labels short (1-3 words)
7. Make sure ALL states mentioned in transitions are listed in states

Examples:

Input: "A traffic light switches from red to green to yellow in order"
Output:
states: red, green, yellow
transitions: red --timer--> green, green --timer--> yellow, yellow --timer--> red

Input: "A vending machine accepts coins until reaching 10 rupees"
Output:
states: no_credit, has_credit, dispensing
transitions: no_credit --coin_inserted--> has_credit, has_credit --sufficient_credit--> dispensing, dispensing --item_taken--> no_credit

Input: "A door lock toggles between locked and unlocked with key"
Output:
states: locked, unlocked
transitions: locked --key_turn--> unlocked, unlocked --key_turn--> locked

Now generate for the given input. Only output the states and transitions lines, nothing else."""
    
    models_to_try = [
        "gemini-2.5-flash",
        "gemini-2.5-pro"
    ]
    
    last_error = None
    
    for model_name in models_to_try:
        try:
            print(f"Attempting to use model: {model_name}")
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            
            result = response.text.strip()
            
            # Clean up the result
            lines = []
            for line in result.split('\n'):
                line = line.strip()
                if line.startswith('states:') or line.startswith('transitions:'):
                    lines.append(line)
            
            cleaned_result = '\n'.join(lines) if lines else result
            print(f"✓ Successfully generated using {model_name}")
            return cleaned_result
            
        except Exception as e:
            error_msg = str(e).lower()
            print(f"✗ Model {model_name} failed: {str(e)}")
            last_error = e
            
            if any(code in error_msg for code in ['503', '429', 'overload', 'rate limit', 'quota', 'resource exhausted']):
                print(f"  → Detected overload/rate limit, trying next model...")
                continue
            else:
                print(f"  → Non-overload error, trying next model anyway...")
                continue
    
    raise Exception(f"All models failed. Last error: {str(last_error)}")


@app.route('/')
def home():
    """Serve the main HTML page"""
    return render_template('index.html')


@app.route('/api/generate', methods=['POST'])
def generate_fsm():
    """API endpoint to generate FSM"""
    try:
        data = request.get_json()
        description = data.get('description', '').strip()
        print(f"Generating FSM for: {description}")
        
        if not description:
            return jsonify({
                'success': False,
                'error': 'Description cannot be empty'
            }), 400
        
        result = generate(description)
        print(f"Generated FSM: {result}")
        
        return jsonify({
            'success': True,
            'input': description,
            'output': result
        })
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to generate FSM: {str(e)}'
        }), 500


@app.route('/api/examples', methods=['GET'])
def get_examples():
    """Get example scenarios"""
    examples = [
        "A traffic light switches from red to green to yellow in order",
        "A vending machine accepts coins until reaching 10 rupees then dispenses item",
        "A door lock toggles between locked and unlocked with key",
        "A fan has three speeds: off, low, medium, high cycling in order",
        "An elevator moves between ground, first, and second floors",
        "A washing machine cycles through wash, rinse, and spin then stops",
        "A microwave starts heating when door closes and stops when timer ends",
        "A turnstile unlocks on token insertion and locks after person passes through",
    ]
    return jsonify({'examples': examples})


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'FSM Generator API is running'})


if __name__ == '__main__':
    if not GEMINI_API_KEY:
        print(" Warning: GEMINI_API_KEY not set!")
    print("\n Server starting")
    print("Local:   http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)