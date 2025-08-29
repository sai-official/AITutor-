# AI Tutor App - Complete Implementation
# Requirements: pip install PyPDF2 nltk ollama streamlit textstat scikit-learn

import streamlit as st
import PyPDF2
import nltk
import ollama
import json
import re
import random
from typing import List, Dict, Tuple
from collections import Counter
from textstat import flesch_reading_ease
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
nltk.download('punkt_tab')
# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('chunkers/maxent_ne_chunker')
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.stem import WordNetLemmatizer

st.set_page_config(
            page_title="AI Tutor Assistant",
            page_icon="🎓",
            layout="wide"
        )
class DocumentProcessor:
    """Handles PDF reading and text processing"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text content from uploaded PDF"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return ""
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        return text.strip()
    
    def extract_key_concepts(self, text: str, num_concepts: int = 20) -> List[str]:
        """Extract key concepts using NLP techniques"""
        # Tokenize and get sentences
        sentences = sent_tokenize(text)
        
        # Use TF-IDF to find important terms
        vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=2
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get average TF-IDF scores
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            concept_scores = list(zip(feature_names, mean_scores))
            concept_scores.sort(key=lambda x: x[1], reverse=True)
            
            return [concept[0] for concept in concept_scores[:num_concepts]]
        except:
            # Fallback: extract noun phrases
            words = word_tokenize(text.lower())
            words = [word for word in words if word.isalpha() and word not in self.stop_words]
            pos_tags = pos_tag(words)
            
            # Extract nouns and adjectives
            important_words = [word for word, pos in pos_tags if pos.startswith(('NN', 'JJ'))]
            word_freq = Counter(important_words)
            
            return [word for word, _ in word_freq.most_common(num_concepts)]

class OllamaAI:
    """Handles all AI operations using Ollama"""
    
    def __init__(self, model_name: str = "llama3.2"):
        self.model_name = model_name
        self.context_limit = 4000  # Approximate context limit
        
    def check_ollama_connection(self) -> bool:
        """Check if Ollama is running and accessible"""
        try:
            response = ollama.list()
            return True
        except Exception as e:
            st.error(f"Ollama connection error: {str(e)}")
            st.error("Please ensure Ollama is installed and running. Visit https://ollama.ai for setup instructions.")
            return False
    
    def chunk_text(self, text: str, max_chunk_size: int = 3000) -> List[str]:
        """Split text into manageable chunks"""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < max_chunk_size:
                current_chunk += " " + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
                
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks
    
    def generate_response(self, prompt: str, context: str = "") -> str:
        """Generate response using Ollama"""
        try:
            full_prompt = f"{context}\n\n{prompt}" if context else prompt
            
            # Truncate if too long
            if len(full_prompt) > self.context_limit:
                full_prompt = full_prompt[:self.context_limit] + "..."
                
            response = ollama.generate(
                model=self.model_name,
                prompt=full_prompt,
                options={
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'max_tokens': 500
                }
            )
            return response['response']
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def summarize_text(self, text: str, summary_type: str = "detailed") -> str:
        """Generate summary of the text"""
        chunks = self.chunk_text(text)
        summaries = []
        
        for chunk in chunks:
            if summary_type == "brief":
                prompt = f"Provide a brief, concise summary (2-3 sentences) of the following text:\n\n{chunk}"
            else:
                prompt = f"Provide a detailed summary highlighting key points and main concepts of the following text:\n\n{chunk}"
            
            summary = self.generate_response(prompt)
            summaries.append(summary)
        
        # Combine summaries if multiple chunks
        if len(summaries) > 1:
            combined_summary = "\n\n".join(summaries)
            final_prompt = f"Combine and synthesize the following summaries into a coherent {summary_type} summary:\n\n{combined_summary}"
            return self.generate_response(final_prompt)
        
        return summaries[0] if summaries else "Unable to generate summary."
    
    def generate_flashcards(self, text: str, num_cards: int = 10) -> List[Dict[str, str]]:
        """Generate flashcards from the text"""
        flashcards = []
        chunks = self.chunk_text(text)
        
        cards_per_chunk = max(1, num_cards // len(chunks))
        
        for chunk in chunks:
            prompt = f"""Create {cards_per_chunk} educational flashcards from the following text. 
            Format each flashcard as:
            Q: [Question]
            A: [Answer]
            
            Focus on key concepts, definitions, important facts, and relationships. Make questions clear and answers concise.
            
            Text: {chunk}"""
            
            response = self.generate_response(prompt)
            
            # Parse the response to extract Q&A pairs
            cards = self._parse_flashcards(response)
            flashcards.extend(cards)
            
        return flashcards[:num_cards]
    
    def generate_quiz(self, text: str, num_questions: int = 5) -> List[Dict]:
        """Generate multiple choice quiz questions"""
        quiz_questions = []
        chunks = self.chunk_text(text)
        
        questions_per_chunk = max(1, num_questions // len(chunks))
        
        for chunk in chunks:
            prompt = f"""Create {questions_per_chunk} multiple choice questions from the following text.
            Format each question as:
            
            Question: [Question text]
            A) [Option A]
            B) [Option B]  
            C) [Option C]
            D) [Option D]
            Correct: [A/B/C/D]
            Explanation: [Brief explanation]
            
            Make questions challenging but fair, testing understanding of key concepts.
            
            Text: {chunk}"""
            
            response = self.generate_response(prompt)
            questions = self._parse_quiz_questions(response)
            quiz_questions.extend(questions)
            
        return quiz_questions[:num_questions]
    
    def answer_question(self, question: str, context: str) -> str:
        """Answer user questions based on document context"""
        prompt = f"""Based on the following document content, answer the user's question. 
        If the answer is not in the document, say so clearly.
        
        Document content: {context}
        
        Question: {question}
        
        Answer:"""
        
        return self.generate_response(prompt)
    
    def _parse_flashcards(self, response: str) -> List[Dict[str, str]]:
        """Parse flashcard response into structured format"""
        flashcards = []
        lines = response.strip().split('\n')
        current_q = ""
        current_a = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith('Q:'):
                current_q = line[2:].strip()
            elif line.startswith('A:'):
                current_a = line[2:].strip()
                if current_q and current_a:
                    flashcards.append({
                        'question': current_q,
                        'answer': current_a
                    })
                    current_q = ""
                    current_a = ""
        
        return flashcards
    
    def _parse_quiz_questions(self, response: str) -> List[Dict]:
        """Parse quiz response into structured format"""
        questions = []
        lines = response.strip().split('\n')
        current_question = {}
        
        for line in lines:
            line = line.strip()
            if line.startswith('Question:'):
                if current_question:
                    questions.append(current_question)
                current_question = {'question': line[9:].strip(), 'options': {}}
            elif line.startswith(('A)', 'B)', 'C)', 'D)')):
                option_key = line[0]
                option_value = line[2:].strip()
                current_question['options'][option_key] = option_value
            elif line.startswith('Correct:'):
                current_question['correct'] = line[8:].strip()
            elif line.startswith('Explanation:'):
                current_question['explanation'] = line[12:].strip()
        
        if current_question:
            questions.append(current_question)
            
        return questions

class AITutorApp:
    """Main application class"""
    
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.ai = OllamaAI()
        
        # Initialize session state
        if 'document_text' not in st.session_state:
            st.session_state.document_text = ""
        if 'key_concepts' not in st.session_state:
            st.session_state.key_concepts = []
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'flashcards' not in st.session_state:
            st.session_state.flashcards = []
        if 'quiz_questions' not in st.session_state:
            st.session_state.quiz_questions = []
        if 'current_flashcard' not in st.session_state:
            st.session_state.current_flashcard = 0
        if 'show_answer' not in st.session_state:
            st.session_state.show_answer = False
    
    def run(self):
        """Main application runner"""
        
        st.title("🎓 AI Tutor Assistant")
        st.markdown("Upload a PDF document and get AI-powered tutoring with summaries, flashcards, quizzes, and Q&A!")
        
        # Check Ollama connection
        if not self.ai.check_ollama_connection():
            st.stop()
        
        # Sidebar for document upload and settings
        with st.sidebar:
            st.header("📄 Document Upload")
            uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
            
            if uploaded_file is not None:
                if st.button("Process Document", type="primary"):
                    with st.spinner("Processing document..."):
                        text = self.doc_processor.extract_text_from_pdf(uploaded_file)
                        if text:
                            st.session_state.document_text = self.doc_processor.preprocess_text(text)
                            st.session_state.key_concepts = self.doc_processor.extract_key_concepts(text)
                            st.success("Document processed successfully!")
                        else:
                            st.error("Failed to extract text from PDF")
            
            # Document info
            if st.session_state.document_text:
                st.header("📊 Document Info")
                word_count = len(st.session_state.document_text.split())
                readability = flesch_reading_ease(st.session_state.document_text)
                
                st.metric("Word Count", word_count)
                st.metric("Readability Score", f"{readability:.1f}")
                
                if st.session_state.key_concepts:
                    st.subheader("🔑 Key Concepts")
                    for concept in st.session_state.key_concepts[:10]:
                        st.text(f"• {concept}")
        
        # Main content area
        if not st.session_state.document_text:
            st.info("👆 Please upload and process a PDF document to get started!")
            return
        
        # Create tabs for different functionalities
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "💬 Chat", "📝 Summary", "🗂️ Flashcards", "🧩 Quiz", "📖 Document"
        ])
        
        with tab1:
            self.render_chat_interface()
            
        with tab2:
            self.render_summary_interface()
            
        with tab3:
            self.render_flashcard_interface()
            
        with tab4:
            self.render_quiz_interface()
            
        with tab5:
            self.render_document_viewer()
    
    # def render_chat_interface(self):
    #     st.subheader("Chat with your AI Tutor")

    #     # Display previous messages
    #     for msg in st.session_state.chat_history:
    #         with st.chat_message(msg["role"]):
    #             st.markdown(msg["content"])

    #     # --- chat_input must be top-level ---
    #     user_question = st.chat_input("Ask a question about the document...")

    #     if user_question:
    #         # Save user message
    #         st.session_state.chat_history.append({"role": "user", "content": user_question})
    #         with st.chat_message("user"):
    #             st.markdown(user_question)

    #         # Generate AI response
    #         response = self.answer_question(user_question)

    #         # Save AI response
    #         st.session_state.chat_history.append({"role": "assistant", "content": response})
    #         with st.chat_message("assistant"):
    #             st.markdown(response)

    
    def render_chat_interface(self):
        """Render the chat interface"""
        st.header("💬 Ask Questions About Your Document")
        
        # Display chat history
        for chat in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(chat["question"])
            with st.chat_message("assistant"):
                st.write(chat["answer"])
        
        # Chat input
        user_question = st.chat_input("Ask a question about the document...")
        
        if user_question:
            # Add user question to chat
            with st.chat_message("user"):
                st.write(user_question)
            
            # Generate AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = self.ai.answer_question(user_question, st.session_state.document_text)
                    st.write(answer)
                    
                    # Save to chat history
                    st.session_state.chat_history.append({
                        "question": user_question,
                        "answer": answer
                    })
        
        # Clear chat button
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

    
    def render_summary_interface(self):
        """Render the summary interface"""
        st.header("📝 Document Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            summary_type = st.selectbox(
                "Summary Type",
                ["detailed", "brief"],
                help="Choose between detailed or brief summary"
            )
        
        with col2:
            if st.button("Generate Summary", type="primary"):
                with st.spinner("Generating summary..."):
                    summary = self.ai.summarize_text(st.session_state.document_text, summary_type)
                    st.session_state.current_summary = summary
        
        # Display summary
        if hasattr(st.session_state, 'current_summary'):
            st.subheader("Summary")
            st.write(st.session_state.current_summary)
            
            # Option to download summary
            st.download_button(
                "Download Summary",
                st.session_state.current_summary,
                file_name="document_summary.txt",
                mime="text/plain"
            )
    
    def render_flashcard_interface(self):
        """Render the flashcard interface"""
        st.header("🗂️ Flashcards")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            num_cards = st.slider("Number of Flashcards", min_value=5, max_value=20, value=10)
        
        with col2:
            if st.button("Generate Flashcards", type="primary"):
                with st.spinner("Creating flashcards..."):
                    flashcards = self.ai.generate_flashcards(st.session_state.document_text, num_cards)
                    st.session_state.flashcards = flashcards
                    st.session_state.current_flashcard = 0
                    st.session_state.show_answer = False
        
        # Display flashcards
        if st.session_state.flashcards:
            total_cards = len(st.session_state.flashcards)
            current_idx = st.session_state.current_flashcard
            
            st.subheader(f"Flashcard {current_idx + 1} of {total_cards}")
            
            # Navigation
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("⬅️ Previous") and current_idx > 0:
                    st.session_state.current_flashcard -= 1
                    st.session_state.show_answer = False
                    st.rerun()
            
            with col2:
                if st.button("➡️ Next") and current_idx < total_cards - 1:
                    st.session_state.current_flashcard += 1
                    st.session_state.show_answer = False
                    st.rerun()
            
            with col3:
                if st.button("🔄 Shuffle"):
                    random.shuffle(st.session_state.flashcards)
                    st.session_state.current_flashcard = 0
                    st.session_state.show_answer = False
                    st.rerun()
            
            with col4:
                if st.button("👁️ Show/Hide Answer"):
                    st.session_state.show_answer = not st.session_state.show_answer
                    st.rerun()
            
            # Display current flashcard
            if st.session_state.flashcards:
                card = st.session_state.flashcards[current_idx]
                
                # Question
                st.markdown("### Question:")
                st.info(card['question'])
                
                # Answer (toggle visibility)
                if st.session_state.show_answer:
                    st.markdown("### Answer:")
                    st.success(card['answer'])
            
            # Download flashcards
            flashcard_text = self._format_flashcards_for_download(st.session_state.flashcards)
            st.download_button(
                "Download All Flashcards",
                flashcard_text,
                file_name="flashcards.txt",
                mime="text/plain"
            )
    
    def render_quiz_interface(self):
        """Render the quiz interface"""
        st.header("🧩 Interactive Quiz")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            num_questions = st.slider("Number of Questions", min_value=3, max_value=15, value=5)
        
        with col2:
            if st.button("Generate Quiz", type="primary"):
                with st.spinner("Creating quiz..."):
                    questions = self.ai.generate_quiz(st.session_state.document_text, num_questions)
                    st.session_state.quiz_questions = questions
                    st.session_state.quiz_answers = {}
                    st.session_state.quiz_submitted = False
        
        # Display quiz
        if st.session_state.quiz_questions:
            if not hasattr(st.session_state, 'quiz_submitted'):
                st.session_state.quiz_submitted = False
            
            with st.form("quiz_form"):
                answers = {}
                
                for i, q in enumerate(st.session_state.quiz_questions):
                    st.subheader(f"Question {i+1}")
                    st.write(q['question'])
                    
                    if 'options' in q and q['options']:
                        options = [f"{k}) {v}" for k, v in q['options'].items()]
                        selected = st.radio(
                            "Select your answer:",
                            options,
                            key=f"q_{i}",
                            index=None
                        )
                        if selected:
                            answers[i] = selected[0]  # Extract A, B, C, or D
                
                submitted = st.form_submit_button("Submit Quiz", type="primary")
                
                if submitted:
                    st.session_state.quiz_answers = answers
                    st.session_state.quiz_submitted = True
                    st.rerun()
            
            # Show results
            if st.session_state.quiz_submitted:
                self._display_quiz_results()
    
    def render_document_viewer(self):
        """Render document viewer"""
        st.header("📖 Document Content")
        
        # Display document statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            word_count = len(st.session_state.document_text.split())
            st.metric("Words", word_count)
        
        with col2:
            sentence_count = len(sent_tokenize(st.session_state.document_text))
            st.metric("Sentences", sentence_count)
        
        with col3:
            char_count = len(st.session_state.document_text)
            st.metric("Characters", char_count)
        
        # Search within document
        search_term = st.text_input("Search in document:")
        
        if search_term:
            # Highlight search terms
            highlighted_text = st.session_state.document_text
            highlighted_text = highlighted_text.replace(
                search_term, 
                f"**{search_term}**"
            )
            st.markdown("### Search Results:")
            st.markdown(highlighted_text)
        else:
            # Display full document
            st.markdown("### Full Document Content:")
            st.text_area(
                "Document Text", 
                st.session_state.document_text, 
                height=400,
                disabled=True
            )
    
    def _format_flashcards_for_download(self, flashcards: List[Dict[str, str]]) -> str:
        """Format flashcards for download"""
        formatted = "AI TUTOR - FLASHCARDS\n" + "="*50 + "\n\n"
        
        for i, card in enumerate(flashcards, 1):
            formatted += f"CARD {i}\n"
            formatted += f"Q: {card['question']}\n"
            formatted += f"A: {card['answer']}\n\n"
            formatted += "-" * 30 + "\n\n"
        
        return formatted
    
    def _display_quiz_results(self):
        """Display quiz results and scoring"""
        st.subheader("📊 Quiz Results")
        
        correct_count = 0
        total_questions = len(st.session_state.quiz_questions)
        
        for i, q in enumerate(st.session_state.quiz_questions):
            user_answer = st.session_state.quiz_answers.get(i)
            correct_answer = q.get('correct', '').strip().upper()
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**Q{i+1}:** {q['question']}")
                
                if user_answer:
                    if user_answer.upper() == correct_answer:
                        st.success(f"✅ Correct! You answered: {user_answer}")
                        correct_count += 1
                    else:
                        st.error(f"❌ Incorrect. You answered: {user_answer}, Correct: {correct_answer}")
                else:
                    st.warning("⚠️ No answer provided")
                
                if 'explanation' in q:
                    st.info(f"💡 Explanation: {q['explanation']}")
            
            st.divider()
        
        # Final score
        score = (correct_count / total_questions) * 100
        st.subheader("Final Score")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Score", f"{score:.1f}%")
        with col2:
            st.metric("Correct", f"{correct_count}/{total_questions}")
        with col3:
            if score >= 80:
                st.success("🌟 Excellent!")
            elif score >= 60:
                st.info("👍 Good job!")
            else:
                st.warning("📚 Keep studying!")

# Streamlit app configuration and styling
def apply_custom_css():
    """Apply custom CSS for better styling"""
    st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
    }
    .flashcard-container {
        border: 2px solid #f0f0f0;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        background-color: #f9f9f9;
    }
    .quiz-question {
        background-color: #e6f3ff;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Main application entry point
def main():
    """Main function to run the app"""
    apply_custom_css()
    
    # Initialize and run the app
    app = AITutorApp()
    app.run()

if __name__ == "__main__":
    main()

# Additional utility functions for enhanced functionality

class AdvancedNLPProcessor:
    """Advanced NLP processing for better content understanding"""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def extract_named_entities(self, text: str) -> List[str]:
        """Extract named entities from text"""
        sentences = sent_tokenize(text)
        entities = []
        
        for sentence in sentences:
            words = word_tokenize(sentence)
            pos_tags = pos_tag(words)
            chunks = ne_chunk(pos_tags)
            
            for chunk in chunks:
                if hasattr(chunk, 'label'):
                    entity = ' '.join([token for token, pos in chunk.leaves()])
                    entities.append(entity)
        
        return list(set(entities))
    
    def extract_definition_pairs(self, text: str) -> List[Tuple[str, str]]:
        """Extract definition pairs from text"""
        definitions = []
        sentences = sent_tokenize(text)
        
        # Patterns that indicate definitions
        definition_patterns = [
            r'(.+?)\s+is\s+(.+?)(?:\.|$)',
            r'(.+?)\s+refers to\s+(.+?)(?:\.|$)', 
            r'(.+?)\s+means\s+(.+?)(?:\.|$)',
            r'(.+?):\s+(.+?)(?:\.|$)'
        ]
        
        for sentence in sentences:
            for pattern in definition_patterns:
                matches = re.findall(pattern, sentence, re.IGNORECASE)
                for match in matches:
                    term = match[0].strip()
                    definition = match[1].strip()
                    if len(term) < 50 and len(definition) > 10:
                        definitions.append((term, definition))
        
        return definitions
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        vectorizer = TfidfVectorizer(stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except:
            return 0.0


