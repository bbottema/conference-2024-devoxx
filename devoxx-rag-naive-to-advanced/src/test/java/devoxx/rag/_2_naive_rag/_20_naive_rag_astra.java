package devoxx.rag._2_naive_rag;

import com.datastax.astra.client.DataAPIClient;
import com.datastax.astra.client.exception.TooManyDocumentsToCountException;
import com.datastax.astra.langchain4j.store.embedding.AstraDbEmbeddingStore;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.TextDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.input.Prompt;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import devoxx.rag.AbstractDevoxxTest;
import devoxx.rag.Assistant;
import devoxx.rag.ExtendedInMemoryEmbeddingStore;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.net.URL;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static com.datastax.astra.internal.utils.AnsiUtils.cyan;
import static com.datastax.astra.internal.utils.AnsiUtils.yellow;
import static java.util.stream.Collectors.joining;

    public class _20_naive_rag_astra extends AbstractDevoxxTest {

    static final String COLLECTION_NAME = "naive_rag";

    @Test
    public void should_ingest_document() throws TooManyDocumentsToCountException {
        System.out.println(yellow("Ingesting a document"));

        // Load the document
        URL fileURL = getClass().getResource("/text/johnny.txt");
        Path path = new File(fileURL.getFile()).toPath();
        Document myDoc = FileSystemDocumentLoader.loadDocument(path, new TextDocumentParser());
        System.out.println(cyan("[OK] ") + " Document found");

        // Embedding Model (could be OpenAI or local implementation)
        EmbeddingModel embeddingModel = getEmbeddingModel();  // Assume this method provides the embedding model
        System.out.println(cyan("[OK] ") + " Embedding Model '" + MODEL_EMBEDDING_TEXT + "' initialized");

        // In-Memory Embedding Store
        ExtendedInMemoryEmbeddingStore<TextSegment> embeddingStore = new ExtendedInMemoryEmbeddingStore<>();
        embeddingStore.removeAll();  // Clear the store before ingesting new data
        System.out.println(cyan("[OK] ") + " In-Memory Embedding Store initialized");

        // Ingest into the In-Memory Embedding Store
        EmbeddingStoreIngestor.builder()
                .documentSplitter(DocumentSplitters.recursive(300, 20))  // Split the document into chunks
                .embeddingModel(embeddingModel)  // Use the embedding model for embedding
                .embeddingStore(embeddingStore)  // Store embeddings in the in-memory store
                .build()
                .ingest(myDoc);  // Perform the ingestion

        System.out.println(cyan("[OK] ") + " Store initialized with " + embeddingStore.getEntries().size() + " elements");
    }

    @Test
    public void should_rag_manual() {
        System.out.println(yellow("RAG with search"));

        String question = "Who is Johnny?";

        // RAG CONTEXT
        ExtendedInMemoryEmbeddingStore<TextSegment> embeddingStore = new ExtendedInMemoryEmbeddingStore<>();
        EmbeddingModel embeddingModel = getEmbeddingModel();
        List<EmbeddingMatch<TextSegment>> relevantEmbeddings = embeddingStore.search(EmbeddingSearchRequest.builder()
                        .queryEmbedding(embeddingModel.embed(question).content())
                        .minScore(0.5)
                        .maxResults(2)
                        .build())
                        .matches();

        // Build Variables
        Map<String, Object> variables = new HashMap<>();
        variables.put("question", question);
        variables.put("rag-context", relevantEmbeddings.stream()
                .map(match -> match.embedded().text())
                .collect(joining("\n\n")));
        System.out.println(variables);


        Prompt prompt = PromptTemplate.from(
                "Answer the following question to the best of your ability:\n"
                        + "\n"
                        + "Question:\n"
                        + "{{question}}\n"
                        + "\n"
                        + "Base your answer on the following information:\n"
                        + "{{rag-context}}").apply(variables);

        // See an answer from the model
        System.out.println(getChatLanguageModel()
                .generate(prompt.toUserMessage())
                .content().text());
    }

    @Test
    public void should_rag_content_retriever() {
        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(new InMemoryEmbeddingStore<>())
                .embeddingModel(getEmbeddingModel())
                .maxResults(2)
                .minScore(0.5)
                .build();

        // configuring it to use the components we've created above.
        Assistant ai = AiServices.builder(Assistant.class)
                .contentRetriever(contentRetriever)
                .chatLanguageModel(getChatLanguageModel())
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .build();

        String response = ai.answer("Who is Johnny?");
        System.out.println(response);
    }

    @Test
    public void deleteCollection() {
        System.out.println(yellow("Delete Collection"));
        ExtendedInMemoryEmbeddingStore<TextSegment> embeddingStore = new ExtendedInMemoryEmbeddingStore<>();
        embeddingStore.removeAll();
        System.out.println(cyan("[OK] ") + " Collection Deleted");
    }
}