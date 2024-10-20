package devoxx.rag;

import com.datastax.astra.client.Collection;
import com.datastax.astra.client.DataAPIClient;
import com.datastax.astra.client.Database;
import com.datastax.astra.client.model.Document;
import com.fasterxml.jackson.databind.ObjectMapper;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.TextDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.chat.StreamingChatLanguageModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.model.openai.OpenAiEmbeddingModel;
import dev.langchain4j.model.openai.OpenAiEmbeddingModel.OpenAiEmbeddingModelBuilder;
import dev.langchain4j.model.openai.OpenAiStreamingChatModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.model.scoring.ScoringModel;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

import static com.datastax.astra.client.model.SimilarityMetric.COSINE;
import static com.datastax.astra.internal.utils.AnsiUtils.cyan;
import static dev.langchain4j.data.document.loader.FileSystemDocumentLoader.loadDocument;

/**
 * Abstract Class for different tests and use cases to share configuration
 */
public abstract class AbstractDevoxxTest {

    // ------------------------------------------------------------
    //                           GEMINI STUFF
    // ------------------------------------------------------------

    // Chat Models
    protected final String MODEL_GEMINI_FLASH = "gemini-1.5-flash";

    // Embedding Models
    protected static final String MODEL_EMBEDDING_MULTILINGUAL = "text-multilingual-embedding-002";
    protected static final String MODEL_EMBEDDING_TEXT         = "text-embedding-004";
    protected static final String MODEL_OPENAI_ADA2            = "text-embedding-ada-002";
    protected static final String MODEL_OPENAI_TEXT3_SMALL     = "text-embedding-3-small";
    protected static final String MODEL_OPENAI_TEXT3_LARGE     = "text-embedding-3-large";
    protected static final String MODEL_OPENAI_GPT4O           = "gpt-4o";
    protected static final String MODEL_OPENAI_GPT35_TURBO     = "gpt-3.5-turbo";
    protected static final int    MODEL_EMBEDDING_DIMENSION    = 768;

    /** Create a chat model. */
    protected ChatLanguageModel getChatLanguageModel() {
        return getChatLanguageModel(MODEL_OPENAI_GPT4O);
    }

    protected ChatLanguageModel getChatLanguageModel(String modelName) {
        return OpenAiChatModel.builder()
                .apiKey(System.getenv("OPENAI_API_KEY"))
                .modelName(modelName)
                .maxRetries(5)
                .build();
    }

    /** Create a streaming chat model. */
    protected StreamingChatLanguageModel getChatLanguageModelStreaming() {
        return OpenAiStreamingChatModel.builder()
                .apiKey(System.getenv("OPENAI_API_KEY"))
                .modelName("gpt-4o") // e.g., "gpt-4"
                .build();
    }

    /** Create an embedding model. */
    protected static EmbeddingModel getEmbeddingModel() {
        return getEmbeddingModel(MODEL_OPENAI_ADA2);
    }
    protected static EmbeddingModel getEmbeddingModel(String modelName) {
        return getEmbeddingModelBuilder(modelName).build();
    }
    protected static OpenAiEmbeddingModelBuilder getEmbeddingModelBuilder(String modelName) {
        return OpenAiEmbeddingModel.builder()
                .apiKey(System.getenv("OPENAI_API_KEY"))
                .modelName(modelName) // This is the common embedding model from OpenAI
                .maxRetries(5);
    }

    // ------------------------------------------------------------
    //                ASTRA / CASSANDRA STORE STUFF
    // ------------------------------------------------------------

    public static final String ASTRA_TOKEN        = System.getenv("ASTRA_TOKEN_DEVOXX");
    public static final String ASTRA_API_ENDPOINT = "https://57fe123e-8f47-4165-babc-0df44136e3fb-us-east1.apps.astra.datastax.com";

    public Database getAstraDatabase() {
        // verbose
        //return new DataAPIClient(ASTRA_TOKEN).getDatabase(ASTRA_API_ENDPOINT);
        return new DataAPIClient(ASTRA_TOKEN)
                //DataAPIOptions.builder().withObserver(new LoggingCommandObserver(AbstractDevoxxTest.class)).build())
                .getDatabase(ASTRA_API_ENDPOINT);
    }

    public Collection<Document> createCollection(String name, int dimension) {
        return getAstraDatabase().createCollection(name, dimension, COSINE);
    }

    public Collection<Document> getCollection(String name) {
        return getAstraDatabase().getCollection(name);
    }

    // ------------------------------------------------------------
    //                JAVA IN-MEMORY STORE STUFF
    // ------------------------------------------------------------

    public InMemoryDatabase getInMemoryDatabase() {
        return new InMemoryDatabase();
    }

    public InMemoryDatabase.Collection createInmemoryCollection(String name, int dimension) {
        return getInMemoryDatabase().createCollection(name, dimension);
    }

    public InMemoryDatabase.Collection getInMemoryCollection(String name) {
        return getInMemoryDatabase().getCollection(name);
    }

    // ------------------------------------------------------------
    //               RAG STUFF
    // ------------------------------------------------------------

    protected void ingestDocument(String docName, EmbeddingModel model, EmbeddingStore<TextSegment> store) {
        Path path = new File(Objects.requireNonNull(AbstractDevoxxTest.class
                .getResource("/" + docName)).getFile()).toPath();
        dev.langchain4j.data.document.Document document = FileSystemDocumentLoader
                .loadDocument(path, new TextDocumentParser());
        DocumentSplitter splitter = DocumentSplitters
                .recursive(300, 20);

        EmbeddingStoreIngestor.builder()
                .documentSplitter(splitter)
                .embeddingModel(model)
                .embeddingStore(store).build().ingest(document);
    }

    protected ContentRetriever createRetriever(String fileName) {
        URL fileURL = getClass().getResource(fileName);
        Path path = new File(fileURL.getFile()).toPath();
        dev.langchain4j.data.document.Document document = FileSystemDocumentLoader
                .loadDocument(path, new TextDocumentParser());
        DocumentSplitter splitter = DocumentSplitters
                .recursive(300, 0);
        List<TextSegment> segments = splitter.split(document);
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(getEmbeddingModel().embedAll(segments).content(), segments);
        return EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(getEmbeddingModel())
                .maxResults(2)
                .minScore(0.6)
                .build();
    }

    protected EmbeddingStoreContentRetriever.EmbeddingStoreContentRetrieverBuilder createRetrieverBuilder(String fileName) {
        List<TextSegment> segments = DocumentSplitters
                .recursive(300, 20)
                .split(loadDocument(new File(Objects.requireNonNull(getClass()
                                .getResource(fileName))
                        .getFile()).toPath(), new TextDocumentParser()));
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(getEmbeddingModel().embedAll(segments).content(), segments);
        return EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(getEmbeddingModel())
                .maxResults(2)
                .minScore(0.6);
    }

    protected ScoringModel getScoringModel() {
        return new BasicEmbeddingModelBasedScoringModel(getEmbeddingModel());
    }

    // ------------------------------------------------------------
    //            DISPLAY STUFF
    // ------------------------------------------------------------

    /**
     * Utilities function to show the results in the console
     *
     * @param response
     *      AI Response
     */
    protected static void prettyPrint(Response<AiMessage> response) {
        System.out.println(cyan("\nRESPONSE TEXT:"));
        System.out.println(response.content().text().replaceAll("\\n", "\n"));
        System.out.println();
        prettyPrintMetadata(response);
    }

    protected static void prettyPrintMetadata(Response<AiMessage> response) {
        System.out.println(cyan("\nRESPONSE METADATA:"));
        if (response.finishReason() != null) {
            System.out.println("Finish Reason : " + cyan(response.finishReason().toString()));
        }
        if (response.tokenUsage() != null) {
            System.out.println("Tokens Input  : " + cyan(String.valueOf(response.tokenUsage().inputTokenCount())));
            System.out.println("Tokens Output : " + cyan(String.valueOf(response.tokenUsage().outputTokenCount())));
            System.out.println("Tokens Total  : " + cyan(String.valueOf(response.tokenUsage().totalTokenCount())));
        }
    }

    public dev.langchain4j.data.document.Document loadDocumentText(String fileName) {
        Path path = new File(Objects.requireNonNull(getClass().getResource("/" + fileName)).getFile()).toPath();
        return FileSystemDocumentLoader.loadDocument(path, new TextDocumentParser());
    }

    @SuppressWarnings("unchecked")
    public  List<Quote> loadQuotes(String filePath) throws IOException {
        URL fileURL = getClass().getResource(filePath);
        File inputFile = new File(fileURL.getFile());
        LinkedHashMap<String, Object> sampleQuotes = new ObjectMapper().readValue(inputFile, LinkedHashMap.class);
        List<Quote> result  = new ArrayList<>();
        AtomicInteger quote_idx = new AtomicInteger(0);
        ((LinkedHashMap<?,?>) sampleQuotes.get("quotes")).forEach((k,v) -> {
            ((ArrayList<?>)v).forEach(q -> {
                Map<String, Object> entry = (Map<String,Object>) q;
                String author = (String) k;//(String) entry.get("author");
                String body = (String) entry.get("body");
                List<String> tags = (List<String>) entry.get("tags");
                String rowId = "q_" + author + "_" + quote_idx.getAndIncrement();
                result.add(new Quote(rowId, author, tags, body));
            });
        });
        return result;
    }

}
