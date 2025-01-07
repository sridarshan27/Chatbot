Cromchat-Educational chatbot


🛠️ Tools and Technologies
	1.Programming Language: Python
	2.NLP Libraries:
		NLTK
		Scikit-learn (TF-IDF Vectorizer and Logistic Regression)
	3.Frontend Framework: Streamlit
	4.Machine Learning: Logistic Regression, BERT-based embeddings
	5.Utilities:
		csv for logging conversations
		datetime for timestamps
		os and json for file handling

  🧠 Methodology
	1.Data Preparation
			Intent data stored in a structured JSON format.
	2.Model Development
			TF-IDF vectorizer converts user inputs into numerical features.
			Logistic Regression model classifies inputs into predefined intents.
	3.Response Mechanism
			Classified intents or matched FAQs return specific responses.
			Fallback mechanism ensures meaningful engagement for unrecognized queries.
	4.Interface and Deployment
			Streamlit provides an interactive web-based interface for user interactions.
			Conversation history logged for analysis and improvement.
🤖Try now:
	Cromchat: https://sridarshan27-chatbot-app-ngboup.streamlit.app/

 Prerequisites
	Python 3.8 or above
		Required Python libraries: nltk, streamlit, scikit-learn
  Description of Key Files
	nlp.py
		Contains the core logic for the chatbot, including intent classification, response generation, and Streamlit interface.
	intents.json
		A structured dataset defining user intents, patterns, and corresponding responses.
	requirements.txt
		A list of dependencies and libraries required to run the chatbot.
	chat_log.csv
		Logs user inputs, chatbot responses, and timestamps during interactions for analysis and improvement. This file is created dynamically during runtime.
	Week 3 Project PPT
		A PowerPoint presentation detailing the project's methodology, tools, and findings.
	assets/
		A directory for storing additional resources like pre-downloaded datasets or NLTK data files.

  📬 Contact
		If you have any questions, feedback, or suggestions, feel free to reach out:
	Author: Sri Darshan S.B
	Email: sridarshan2709@gmail.com
	GitHub: sridarshan27
We appreciate your interest in CareBuddy and look forward to hearing from you!

	
 
