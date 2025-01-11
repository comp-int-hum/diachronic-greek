import os.path
from steamroller import Environment

vars = Variables("custom.py")
vars.AddVariables(
    ("OUTPUT_WIDTH", "", 5000),
    ("RANDOM_SEED", "", 0),
    ("PERSEUS_ROOT", "", os.path.expanduser("~/corpora/perseus")),
    ("IOWA_ROOT", "", os.path.expanduser("~/corpora/iowa_greek")),
    ("IOWA_SPREADSHEET", "", "${IOWA_ROOT}/Greek Corpus with Dates PCD 12 19.xlsx"),
    ("IOWA_NT", "", "${IOWA_ROOT}/New Testament Apocrypha.zip"),
    ("IOWA_OT", "", "${IOWA_ROOT}/Old Testament Pseudepigrapha.zip"),
    ("MAX_SUBDOC_LENGTH", "", 100),
    ("MIN_WORD_COUNT", "", 10),
    ("MAX_WORD_PROPORTION", "", 0.7),
    ("TOP_NEIGHBORS", "", 15),
    ("TOP_TOPIC_WORDS", "", 5),
    ("TOPIC_COUNT", "", 50),
    ("MAX_EPOCHS", "", 100),
    ("CUDA_DEVICE", "", "cpu"),
    ("BATCH_SIZE", "", 512),
    ("WINDOW_SIZE", "", 50),
    ("LEARNING_RATE", "", 0.0001),
    ("MIN_TIME", "", None),
    ("MAX_TIME", "", None),
    ("LOWERCASE", "", True),
    ("TIME_FIELD", "", "year"),
    ("CONTENT_FIELD", "", "content"),
    ("AUTHOR_FIELD", "", None),
    ("TITLE_FIELD", "", None),
    ("SPLIT_FIELD", "", "year"),
    (
        "TEST_PROPORTION",
        "If set to a number between 0 and 1, that much of the data will be set aside for test, otherwise all is used for train+val",
        None
    ),
    (
        "VAL_PROPORTION",
        "This proportion of *non-test* data is used for validation (early stop etc)",
        0.1
    )
)

env = Environment(
    variables=vars,
    BUILDERS={
        "MarshallData" : Builder(
            action="python scripts/marshall_data.py --perseus_root ${PERSEUS_ROOT} --iowa_spreadsheet ${SOURCES[0]} --iowa_nt ${SOURCES[1]} --iowa_ot ${SOURCES[2]} --output ${TARGETS[0]}"
        ),
        "SplitData" : Builder(
            action="python scripts/split_data.py --input ${SOURCES[0]} --first_output ${TARGETS[0]} --second_output ${TARGETS[1]} --second_proportion ${PROPORTION} --split_field ${SPLIT_FIELD} --random_seed ${RANDOM_SEED}"
        ),
        "TrainEmbeddings" : Builder(
            action="python scripts/train_embeddings.py --input ${SOURCES[0]} --output ${TARGETS[0]} ${'--lowercase' if LOWERCASE else ''} --max_subdoc_length ${MAX_SUBDOC_LENGTH} --content_field ${CONTENT_FIELD}"
        ),
        "TrainDETM" : Builder(
            action="python scripts/train_detm.py --train ${SOURCES[0]} ${'--val ' + SOURCES[2].rstr() if len(SOURCES) == 3 else ''} --embeddings ${SOURCES[1]} --window_size ${WINDOW_SIZE} --max_subdoc_length ${MAX_SUBDOC_LENGTH} --device ${CUDA_DEVICE} --batch_size ${BATCH_SIZE} --max_epochs ${MAX_EPOCHS} --output ${TARGETS[0]} --num_topics ${TOPIC_COUNT} --learning_rate ${LEARNING_RATE} --min_word_count ${MIN_WORD_COUNT} --max_word_proportion ${MAX_WORD_PROPORTION} ${'--lowercase' if LOWERCASE else ''} --content_field ${CONTENT_FIELD} --time_field ${TIME_FIELD}"
        ),
        "ApplyDETM" : Builder(
            action="python scripts/apply_detm.py --model ${SOURCES[0]} --input ${SOURCES[1]} --max_subdoc_length ${MAX_SUBDOC_LENGTH} --device ${CUDA_DEVICE} --time_field ${TIME_FIELD} --content_field ${CONTENT_FIELD} ${'--lowercase' if LOWERCASE else ''} --output ${TARGETS[0]}"
        ),
        "GenerateWordSimilarityTable" : Builder(
            action="python scripts/generate_word_similarity_table.py --embeddings ${SOURCES[0]} --output ${TARGETS[0]} --target_words ${WORD_SIMILARITY_TARGETS} --top_neighbors ${TOP_NEIGHBORS} ${'--language_code ' + LANGUAGE_CODE if LANGUAGE_CODE else ''}"
        ),
        "CreateMatrices" : Builder(
            action="python scripts/create_matrices.py --topic_annotations ${SOURCES[0]} --output ${TARGETS[0]} --window_size ${WINDOW_SIZE}"
        ),
        "CreateFigures" : Builder(
            action="python scripts/create_figures.py --input ${SOURCES[0]} --word_image ${TARGETS[0]} --temporal_image ${TARGETS[1]} --author_image ${TARGETS[2]} --word_detail_image ${TARGETS[3]} --author_histogram ${TARGETS[4]} --latex ${TARGETS[5]}"
        )
    }
)

data = env.MarshallData(
    "work/data.jsonl.gz",
    [
        env["IOWA_SPREADSHEET"],
        env["IOWA_NT"],
        env["IOWA_OT"]
    ]
)

train_val_data, test_data = env.SplitData(
    [
        "work/train_val_data.jsonl.gz",
        "work/test_data.jsonl.gz"
    ],
    data,
    PROPORTION=env["TEST_PROPORTION"]
) if env.get("TEST_PROPORTION") else (data, None)

embeddings = env.TrainEmbeddings(
    [
        "work/embeddings.bin",
        "work/embeddings.bin.syn1neg.npy",
        "work/embeddings.bin.wv.vectors.npy"
    ],
    train_val_data
)

topic_model = env.TrainDETM(
    "work/detm_model_${MAX_SUBDOC_LENGTH}.bin.gz",
    [
        train_val_data,
        embeddings
    ]
)

word_similarity_table = env.GenerateWordSimilarityTable(
    "work/word_similarity.tex",
    embeddings,
    WORD_SIMILARITY_TARGETS=["ἀγάπη", "ἔρως", "κύων"],
    TOP_NEIGHBORS=5,
)


#labeled = env.ApplyDETM(
#    "work/labeled.jsonl.gz",
#    [topic_model, test_data if test_data else train_val_data]
#)

#matrices = env.CreateMatrices(
#    "work/matrices.pkl.gz",
#    labeled,
#    MAX_SUBDOC_LENGTH=500,
#    WINDOW_SIZE=500
#)
    
figures = env.CreateFigures(
    [
        "work/words.png",
        "work/temporal.png",
        "work/author.png",
        "work/word_detail.png",
        "work/author_histogram.png",
        "work/tables.tex"
    ],
    topic_model,
)
