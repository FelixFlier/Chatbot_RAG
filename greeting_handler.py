import numpy as np
from typing import Dict, Any

class GreetingRecognizer:
    def __init__(self):
        self.greeting_patterns = {
            'formal': [
                'guten morgen', 'guten tag', 'guten abend',
                'sehr geehrte', 'sehr geehrter'
            ],
            'informal': [
                'hi', 'hallo', 'hey', 'servus', 'moin', 'grüß gott',
                'grüß dich', 'grüezi'
            ],
            'query': [
                'wie geht es dir', 'wie geht\'s', 'wie gehts',
                'alles klar', 'was gibt\'s neues'
            ]
        }

    def analyze_greeting(self, text: str) -> Dict[str, Any]:
        text_lower = text.lower().strip()
        
        for greeting_type, patterns in self.greeting_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                return {
                    'is_greeting': True,
                    'type': greeting_type,
                    'confidence': 0.9
                }
        
        return {
            'is_greeting': False,
            'type': None,
            'confidence': 0.0
        }

class GreetingResponder:
    def __init__(self):
        self.responses = {
            'formal': [
                "Ich freue mich auf Ihre Fragen zur Wirtschaftsprüfung. Wie kann ich Ihnen weiterhelfen?",
                "Womit kann ich Ihnen im Bereich Wirtschaftsprüfung behilflich sein?"
            ],
            'informal': [
                "Schön, dass Sie vorbeischauen! Welche Fragen zur Wirtschaftsprüfung haben Sie?",
                "Was möchten Sie über Wirtschaftsprüfung wissen?"
            ],
            'query': [
                "Mir geht es gut! Was möchten Sie über Wirtschaftsprüfung erfahren?",
                "Bestens, danke! Welche Fragen zur Wirtschaftsprüfung kann ich für Sie beantworten?"
            ]
        }

    def get_response(self, greeting_type: str) -> str:
        if greeting_type in self.responses:
            return np.random.choice(self.responses[greeting_type])
        return "Was möchten Sie über Wirtschaftsprüfung wissen?"