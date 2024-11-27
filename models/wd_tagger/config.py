MODELS = {
    'wd-swinv2-v3': {
        'repo_id': "SmilingWolf/wd-swinv2-tagger-v3",
        'target_size': 448
    },
    'wd-convnext-v3': {
        'repo_id': "SmilingWolf/wd-convnext-tagger-v3",
        'target_size': 448
    },
    'wd-vit-v3': {
        'repo_id': "SmilingWolf/wd-vit-tagger-v3",
        'target_size': 448
    },
    'wd-vit-large-v3': {
        'repo_id': "SmilingWolf/wd-vit-large-tagger-v3",
        'target_size': 448
    },
    'wd-eva02-large-v3': {
        'repo_id': "SmilingWolf/wd-eva02-large-tagger-v3",
        'target_size': 448
    }
}

RATING_CATEGORIES = [9]
GENERAL_CATEGORIES = [0]
CHARACTER_CATEGORIES = [4]

DEFAULT_GENERAL_THRESHOLD = 0.35
DEFAULT_CHARACTER_THRESHOLD = 0.85