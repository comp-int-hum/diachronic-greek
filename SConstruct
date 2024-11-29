import os.path
from steamroller import Environment

vars = Variables("custom.py")
vars.AddVariables(
    ("OUTPUT_WIDTH", "", 5000),
    ("RANDOM_SEED", "", 0),
    ("PERSEUS_ROOT", "", os.path.expanduser("~/corpora/perseus")),
    ("MAX_SUBDOC_LENGTHS", "", [500]),
    ("MIN_WORD_OCCURRENCE", "", 10),
    ("MAX_WORD_PROPORTION", "", 0.7),
    ("TOP_NEIGHBORS", "", 15),
    ("TOP_TOPIC_WORDS", "", 5),
    ("TOPIC_COUNT", "", 50),
    ("EPOCHS", "", 500),
    ("CUDA_DEVICE", "", "cpu"),
    ("BATCH_SIZE", "", 1000),
    ("WINDOW_SIZE", "", 50),
    ("LEARNING_RATE", "", 0.0001),
    ("MIN_TIME", "", None),
    ("MAX_TIME", "", None)
)

env = Environment(
    variables=vars,
    BUILDERS={
        "MarshallPerseus" : Builder(
            action="python scripts/marshall_perseus.py --perseus_root ${PERSEUS_ROOT} --tsv_output ${TARGETS[1]} --output ${TARGETS[0]}"
        ),
        "TrainEmbeddings" : Builder(
            action="python scripts/train_embeddings.py --input ${SOURCES[0]} --output ${TARGETS[0]}"
        ),
        "TrainDETM" : Builder(
            action="python scripts/train_detm.py --train ${SOURCES[0]} ${'--val ' + SOURCES[2].rstr() if len(SOURCES) == 3 else ''} --embeddings ${SOURCES[1]} --window_size ${WINDOW_SIZE} --max_subdoc_length ${MAX_SUBDOC_LENGTH} --device ${CUDA_DEVICE} --batch_size ${BATCH_SIZE} --epochs ${EPOCHS} --output ${TARGETS[0]} --num_topics ${TOPIC_COUNT} --learning_rate ${LEARNING_RATE} --min_word_occurrence ${MIN_WORD_OCCURRENCE} --max_word_proportion ${MAX_WORD_PROPORTION}"
        ),
        "ApplyDETM" : Builder(
            action="python scripts/apply_detm.py --model ${SOURCES[0]} --input ${SOURCES[1]} --max_subdoc_length ${MAX_SUBDOC_LENGTH} --device ${CUDA_DEVICE} --output ${TARGETS[0]}"
        ),
        "GenerateWordSimilarityTable" : Builder(
            action="python scripts/generate_word_similarity_table.py --embeddings ${SOURCES[0]} --output ${TARGETS[0]} --target_words ${WORD_SIMILARITY_TARGETS} --top_neighbors ${TOP_NEIGHBORS}"
        ),
        "CreateMatrices" : Builder(
            action="python scripts/create_matrices.py --topic_annotations ${SOURCES[0]} --output ${TARGETS[0]} --window_size ${WINDOW_SIZE}"
        ),
        "CreateFigures" : Builder(
            action="python scripts/create_figures.py --input ${SOURCES[0]} --word_image ${TARGETS[0]} --temporal_image ${TARGETS[1]} --author_image ${TARGETS[2]} --word_detail_image ${TARGETS[3]} --author_histogram ${TARGETS[4]} --latex ${TARGETS[5]}"
        )
    }
)

perseus_documents, author_sheet = env.MarshallPerseus(
    [
        "work/perseus.jsonl.gz",
        "work/sheet.tsv"
    ],
    []
)

embeddings = env.TrainEmbeddings(
    "work/embeddings.bin",
    perseus_documents
)

topic_model = env.TrainDETM(
    "work/detm_model_${MAX_SUBDOC_LENGTH}.bin.gz",
    [
        perseus_documents,
        embeddings
    ],
    BATCH_SIZE=32,
    EPOCHS=50,
    MIN_WORD_OCCURRENCE=1,
    MAX_WORD_PROPORTION=1.0,
    WINDOW_SIZE=500,
    LEARNING_RATE=0.0008*20,
    MAX_SUBDOC_LENGTH=500
)

word_similarity_table = env.GenerateWordSimilarityTable(
    "work/word_similarity.tex",
    embeddings,
    WORD_SIMILARITY_TARGETS=["ἀγάπη", "ἔρως", "κύων"],
    TOP_NEIGHBORS=5
)

labeled = env.ApplyDETM(
    "work/labeled.jsonl.gz",
    [topic_model, perseus_documents],
    MAX_SUBDOC_LENGTH=500
)

matrices = env.CreateMatrices(
    "work/matrices.pkl.gz",
    labeled,
    MAX_SUBDOC_LENGTH=500,
    WINDOW_SIZE=500
)
    
figures = env.CreateFigures(
    [
        "work/words.png",
        "work/temporal.png",
        "work/author.png",
        "work/word_detail.png",
        "work/author_histogram.png",
        "work/tables.tex"
    ],
    matrices,
)
