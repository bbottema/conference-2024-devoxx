package devoxx.rag._3_advanced_rag_ingestion;

import com.google.gson.Gson;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.document.splitter.DocumentByParagraphSplitter;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.model.scoring.ScoringModel;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import devoxx.rag.AbstractDevoxxTest;
import devoxx.rag.ExtendedInMemoryEmbeddingStore;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.Collectors;

import static com.datastax.astra.internal.utils.AnsiUtils.*;

public class _37_hypothetical_questions_embedding extends AbstractDevoxxTest {

    private static final Gson gson = new Gson();
    private static final String PARAGRAPH_KEY = "paragraph";
    static final File BERLIN_HYPOTHETICAL_QUESTIONS_STORE = new File("src/test/resources/berlin_hypothetical_questions_store.json");

    public static ExtendedInMemoryEmbeddingStore getBerlinHypotheticalQuestionsEmbeddingStore() {
        return ExtendedInMemoryEmbeddingStore.init(BERLIN_HYPOTHETICAL_QUESTIONS_STORE);
    }

    private static final ExtendedInMemoryEmbeddingStore DATABASE = getBerlinHypotheticalQuestionsEmbeddingStore();

    @Order(2)
//    @Test
    public void deleteCollection() {
        DATABASE.removeAll();
        DATABASE.serializeToFile(BERLIN_HYPOTHETICAL_QUESTIONS_STORE.toPath());
    }

    @Order(1)
    @Test
    public void ingestionOfHypotheticalQuestions() {
        if (DATABASE.isEmpty()) {
            System.out.println(cyan("Ingesting hypothetical questions..."));

            Document documentAboutBerlin = loadDocumentText("text/berlin.txt");

            ChatLanguageModel chatLanguageModel = getChatLanguageModel(MODEL_OPENAI_GPT35_TURBO);

            // ========================================================
            // Prepare 10 questions for each paragraph thanks to an LLM

            List<QuestionParagraph> allQuestionParagraphs = new ArrayList<>();

            DocumentByParagraphSplitter splitter = new DocumentByParagraphSplitter(2000, 100);
            List<TextSegment> paragraphs = splitter.split(documentAboutBerlin);

            try (ForkJoinPool pool = new ForkJoinPool(3)) {
                pool.submit(() ->
                        paragraphs.parallelStream().forEach(paragraphSegment -> {
                            System.out.println(cyan("\n==== PARAGRAPH ==================================\n") + paragraphSegment.text());

                            Response<AiMessage> aiResult = chatLanguageModel.generate(List.of(
                                    SystemMessage.from("""
                                            Suggest 10 clear questions whose answer could be given by the user provided text.
                                            Don't use pronouns, be explicit about the subjects and objects of the question.
                                            
                                            Important: Return the answer as JSON Array, strip newlines and omit any surrounding markup such as "```".
                                            Example: ["What is the capital of France?", "How many people live in Paris?", ...]
                                            """),
                                    UserMessage.from(paragraphSegment.text())
                            ));

                            String[] questions = gson.fromJson(aiResult.content().text(), String[].class);

                            System.out.println(yellow("\nQUESTIONS:\n"));
                            for (int i = 0; i < questions.length; i++) {
                                String question = questions[i];
                                System.out.println((i + 1) + ") " + question);

                                allQuestionParagraphs.add(new QuestionParagraph(question, paragraphSegment));
                            }
                        })
                ).join();
            }

            // ===============================================
            // Embed all the pairs of questions and paragraphs

            List<TextSegment> embeddedSegments = allQuestionParagraphs.stream()
                    .map(questionParagraph -> TextSegment.from(
                            questionParagraph.question(),
                            new Metadata().put(PARAGRAPH_KEY, questionParagraph.paragraph().text())))
                    .toList();

            var embeddingModel = getEmbeddingModel();
            var embeddingStore = getBerlinHypotheticalQuestionsEmbeddingStore();

            List<Embedding> embeddings = embeddingModel.embedAll(embeddedSegments).content();
            embeddingStore.addAll(embeddings, embeddedSegments);
            embeddingStore.serializeToFile(BERLIN_HYPOTHETICAL_QUESTIONS_STORE.toPath());
        }
    }

    @Order(2)
    @Test
    public void hypotheticalQuestions() {
        // =========================================================
        // Search against the embedded questions, not the paragraphs

        String queryString = "How many inhabitants live in Berlin?";

        System.out.println(magenta("-".repeat(100)));
        System.out.println(magenta("\nUSER QUESTION: ") + queryString);

        var embeddingModel = getEmbeddingModel();
        var embeddingStore = getBerlinHypotheticalQuestionsEmbeddingStore();

        EmbeddingSearchResult<TextSegment> searchResults = embeddingStore.search(EmbeddingSearchRequest.builder()
                .maxResults(4)
                .minScore(0.7)
                .queryEmbedding(embeddingModel.embed(queryString).content())
                .build());

        ScoringModel scoringModel = getScoringModel();

        searchResults.matches().forEach(match -> {
            double score = scoringModel.score(match.embedded().metadata().getString(PARAGRAPH_KEY), queryString).content();

            System.out.println(yellow("\n-> Similarity: " + match.score() + " --- (Ranking score: " + score + ") ---\n") +
                    "\n" + cyan("Embedded question: ") + match.embedded().text() +
                    "\n" + cyan("  About paragraph: ") + match.embedded().metadata().getString(PARAGRAPH_KEY));
        });

        // =================================
        // Ask Gemini to generate a response

        ChatLanguageModel chatModel = getChatLanguageModel();

        String concatenatedExtracts = searchResults.matches().stream()
                .map(match -> match.embedded().metadata().getString(PARAGRAPH_KEY))
                .distinct()
                .collect(Collectors.joining("\n---\n", "\n---\n", "\n---\n"));

        UserMessage userMessage = PromptTemplate.from("""
                You must answer the following question:
                
                {{question}}
                
                Base your answer on the following documentation extracts:
                
                {{extracts}}
                """).apply(Map.of(
                "question", queryString,
                "extracts", concatenatedExtracts
        )).toUserMessage();

        System.out.println(magenta("\nMODEL REQUEST:\n") + userMessage.singleText().replaceAll("\\n", "\n") + "\n");

        Response<AiMessage> response = chatModel.generate(userMessage);

        System.out.println(magenta("\nRESPONSE:\n") + response.content().text());
    }

    record QuestionParagraph(String question, TextSegment paragraph) {
    }
}