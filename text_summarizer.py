import nltk
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from summa import summarizer
from transformers import BartForConditionalGeneration, BartTokenizer
from rouge_score import rouge_scorer
import warnings
warnings.filterwarnings("ignore")

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

class TextSummarizer:
    def __init__(self):
        """Initialize the summarizer with BART model and tokenizer."""
        self.model_name = "facebook/bart-large-cnn"
        self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
        self.model = BartForConditionalGeneration.from_pretrained(self.model_name)
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text):
        """Preprocess text: clean, tokenize, and remove stopwords."""
        # Remove special characters and extra spaces
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.]', '', text)
        
        # Tokenize sentences and words
        sentences = sent_tokenize(text)
        words = [word_tokenize(sentence.lower()) for sentence in sentences]
        
        # Remove stopwords
        cleaned_sentences = [
            ' '.join([word for word in sentence if word not in self.stop_words])
            for sentence in words
        ]
        return sentences, cleaned_sentences, text.strip()

    def extractive_summary(self, text, ratio=0.3):
        """Generate extractive summary using TextRank."""
        return summarizer.summarize(text, ratio=ratio)

    def abstractive_summary(self, text, max_length=150, min_length=50):
        """Generate abstractive summary using BART."""
        inputs = self.tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)
        summary_ids = self.model.generate(
            inputs["input_ids"],
            num_beams=4,
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            early_stopping=True
        )
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    def evaluate_summary(self, generated_summary, reference_summary):
        """Evaluate summary using ROUGE scores."""
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference_summary, generated_summary)
        return scores

    def summarize(self, text, method="both", reference_summary=None):
        """Generate summary based on specified method and evaluate if reference provided."""
        # Preprocess text
        original_sentences, _, cleaned_text = self.preprocess_text(text)
        
        results = {}
        
        # Extractive summary
        if method in ["extractive", "both"]:
            results["extractive"] = self.extractive_summary(cleaned_text)
        
        # Abstractive summary
        if method in ["abstractive", "both"]:
            results["abstractive"] = self.abstractive_summary(cleaned_text)
        
        # Evaluate summaries if reference is provided
        if reference_summary:
            results["evaluation"] = {}
            if "extractive" in results:
                results["evaluation"]["extractive"] = self.evaluate_summary(
                    results["extractive"], reference_summary
                )
            if "abstractive" in results:
                results["evaluation"]["abstractive"] = self.evaluate_summary(
                    results["abstractive"], reference_summary
                )
        
        return results

# Example usage
if __name__ == "__main__":
    sample_text = """
    The rapid advancement of artificial intelligence has transformed industries worldwide. 
    Machine learning algorithms, particularly deep learning models, have achieved remarkable 
    success in tasks such as image recognition, natural language processing, and autonomous 
    driving. However, these advancements come with challenges, including ethical concerns, 
    data privacy issues, and the need for large computational resources. Researchers are 
    actively working on solutions to address these problems, such as developing more efficient 
    algorithms and robust privacy-preserving techniques. The future of AI holds immense 
    potential, but responsible development and deployment are crucial for sustainable growth.
    """
    
    reference_summary = """
    Artificial intelligence, driven by machine learning and deep learning, has revolutionized 
    industries with successes in image recognition, NLP, and autonomous driving. Challenges 
    like ethical concerns, data privacy, and high computational demands persist, but researchers 
    are developing efficient algorithms and privacy solutions. Responsible AI development is 
    essential for its sustainable future.
    """
    
    summarizer = TextSummarizer()
    results = summarizer.summarize(sample_text, method="both", reference_summary=reference_summary)
    
    print("Extractive Summary:")
    print(results["extractive"])
    print("\nAbstractive Summary:")
    print(results["abstractive"])
    print("\nEvaluation Scores:")
    for method, scores in results["evaluation"].items():
        print(f"\n{method.capitalize()} Summary Scores:")
        for metric, score in scores.items():
            print(f"{metric}: Precision={score.precision:.3f}, Recall={score.recall:.3f}, F1={score.fmeasure:.3f}")