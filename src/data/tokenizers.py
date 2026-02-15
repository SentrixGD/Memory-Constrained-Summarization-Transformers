"""
Script: data_cleaning_pipeline.py
Purpose: Clean text articles, compute statistics, generate histograms, and save cleaned datasets.
Inputs: raw train/validation/test CSV files
Outputs: cleaned CSVs and descriptive statistics/histograms in ../data/stats
Dependencies: pandas, re, os, tqdm, matplotlib, numpy, collections, string, unicodedata
"""
import pandas as pd
import re
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import string
import unicodedata

tqdm.pandas(ncols = 100)


def remove_author(text: str) -> str:
    """
    Remove author byline patterns like 'By . Author Name .' at the start of the text.

    Args:
        text (str): The input text string.

    Returns:
        str: Text with author byline removed.
    """
    return re.sub(r'^By\s+\.\s+.*?\.\s+', "", text)


def remove_cnn_markers(text: str) -> str:
    """
    Remove CNN-specific markers like '(CNN) --' and replace double dashes with single dash.

    Args:
        text (str): The input text string.

    Returns:
        str: Text with CNN markers cleaned.
    """
    text = re.sub(r'\(CNN\)\s+--\s+', "", text)
    text = text.replace('--', '-')
    return text


def remove_published(text: str) -> str:
    """
    Remove lines starting with 'PUBLISHED:' and the following content until the newline.

    Args:
        text (str): The input text string.

    Returns:
        str: Text with publication lines removed.
    """
    HEADER_RE = re.compile(r"""
    ^
    (?:By\s*\.\s*.*?\.\s*)?      # optional author block
    PUBLISHED:\s*\.\s*.*?\.\s*   # published metadata
    (?:\|\s*\.\s*)?              # optional separator
    UPDATED:\s*\.\s*.*?\.\s*     # updated metadata
    """, re.VERBOSE)

    text = HEADER_RE.sub("", text).lstrip()
    return text


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in the text: collapse multiple spaces/tabs/newlines to single space.

    Args:
        text (str): The input text string.

    Returns:
        str: Text with normalized whitespace.
    """
    return ' '.join(text.split())


def fix_punctuation(text: str) -> str:
    """
    Fix common punctuation spacing issues:
      - Remove space before punctuation (.,;!?)
      - Remove space after '(' and before ')'
      - Collapse multiple periods to a single period

    Args:
        text (str): The input text string.

    Returns:
        str: Text with cleaned punctuation.
    """
    text = re.sub(r'\s+([.,;!?])', r'\1', text)
    text = re.sub(r'\(\s+', '(', text)
    text = re.sub(r'\s+\)', ')', text)
    text = re.sub(r'\.{2,}', '.', text)
    return text


def normalize_quotes(text: str) -> str:
    """
    Standardize quotes:
      - Convert curly quotes to straight quotes
      - Convert curly apostrophes to straight apostrophes

    Args:
        text (str): The input text string.

    Returns:
        str: Text with normalized quotes.
    """
    text = text.replace('“', '"').replace('”', '"').replace('’', "'")
    return text


def remove_non_english(text: str) -> str:
    """
    Remove special characters:
      - delete chinese characters, emojis, characters with accents and other special characters
      - only english characters, digits and punctuation are the remaining symbols

    Args:
        text (str): The input text string.

    Returns:
        str: Text without special symbols
    """
    return ALLOWED_CHARS.sub("", text)


def ascii_fold(text: str) -> str:
    """
    Normalize text to ASCII by removing diacritics and special characters.
      - Converts accented characters to their closest ASCII equivalent
      - Removes any non-ASCII symbols

    Args:
        text (str): The input text string.

    Returns:
        str: ASCII-normalized text.
    """
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def track_ascii_fold(text: str) -> str:
    """
    Apply ASCII folding to text and track the number of characters removed.
      - Uses `ascii_fold` to normalize text
      - Updates the global counter `total_augmented_chars` to reflect the number of characters changed/removed

    Args:
        text (str): The input text string.

    Returns:
        str: ASCII-folded text
    """
    global total_augmented_chars
    folded = ascii_fold(text)
    total_augmented_chars += len(text) - len(folded)
    return folded


def cleaning(text_input: str) -> str:
    """
    Full cleaning pipeline that applies all cleaning steps sequentially.

    Steps:
      1. Remove author byline
      2. Remove CNN markers
      3. Remove 'PUBLISHED:' lines
      4. Normalize whitespace
      5. Fix punctuation
      6. Normalize quotes

    Args:
        text_input (str): Raw input text.

    Returns:
        str: Fully cleaned text.
    """
    text = remove_author(text_input)
    text = remove_published(text)
    text = remove_cnn_markers(text)
    text = ascii_fold(text)
    text = remove_non_english(text)
    text = normalize_whitespace(text)
    text = fix_punctuation(text)
    text = normalize_quotes(text)
    return text


def build_vocab(series: pd.Series) -> Counter:
    """
    Build a character-level vocabulary from a series of text.
      - Counts occurrences of each character in the series
      - Uses tqdm for progress visualization

    Args:
        series (pd.Series): Series of strings to process

    Returns:
        collections.Counter: Mapping from character to its frequency in the dataset
    """
    counter = Counter()
    for text in tqdm(series, total = len(series)):
        counter.update(list(text))
    return counter


def cleaning_stats(raw_series: pd.Series, cleaned_series: pd.Series) -> dict:
    """
    Compute statistics on characters removed during cleaning.
      - Total raw characters
      - Total characters after cleaning
      - Number of characters removed
      - Percentage of characters removed

    Args:
        raw_series (pd.Series): Series of raw text
        cleaned_series (pd.Series): Series of cleaned text

    Returns:
        dict: Dictionary with keys:
            - "total_raw_chars"
            - "total_cleaned_chars"
            - "chars_removed"
            - "pct_removed"
    """
    stats = {}
    raw_chars = raw_series.str.len().sum()
    cleaned_chars = cleaned_series.str.len().sum()
    stats["total_raw_chars"] = raw_chars
    stats["total_cleaned_chars"] = cleaned_chars
    stats["chars_removed"] = raw_chars - cleaned_chars
    stats["pct_removed"] = 100 * (raw_chars - cleaned_chars) / raw_chars
    return stats


def count_patterns(series: pd.Series) -> dict:
    """
    Count occurrences of predefined patterns across a text series.
      - Uses a global `patterns` dictionary mapping pattern names to regexes
      - Returns total counts for each pattern in the series

    Args:
        series (pd.Series): Series of strings to analyze

    Returns:
        dict: Mapping of pattern name to total occurrence count
    """
    results = {}
    for name, pat in patterns.items():
        results[name] = series.str.count(pat).sum()
    return results


# load the raw data
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw_path = os.path.join(ROOT_DIR, 'data', 'raw')
processed_path = os.path.join(ROOT_DIR, 'data', 'processed')
train = pd.read_csv(os.path.join(raw_path, 'train.csv'))
val = pd.read_csv(os.path.join(raw_path, 'validation.csv'))
test = pd.read_csv(os.path.join(raw_path, 'test.csv'))
ALLOWED_CHARS = re.compile(
    rf"[^A-Za-z0-9\s{re.escape(string.punctuation)}]"
)
patterns = {
    "double_hyphen": r"--",
    "cnn_marker": r"\(CNN\)",
    "published": r"PUBLISHED:",
    "curly_quotes": r"[“”‘’]",
}

train['highlights'] = train['highlights'].str.replace('\n', ' ', regex = False)
val['highlights'] = val['highlights'].str.replace('\n', ' ', regex = False)
# analyze the length of the raw data
train['char_len'] = train['article'].str.len()
pre_clean_char_len = train['char_len']
train['highlight_len'] = train['highlights'].str.len()
train['highlight_ratio'] = train['highlight_len'] / train['char_len']
plt.hist(train['char_len'], bins = 50)
plt.title("Distribution of Article Character Lengths (Before Cleaning)")
plt.savefig("../data/stats/char_len_hist_before_cleaning.png")
plt.close()

# save descriptive statistics for pre-cleaned data to CSV for later analysis
pre_cleaning = train[['char_len', 'highlight_len', 'highlight_ratio']].describe()
pre_cleaning.to_csv("../data/stats/length_stats_pre_cleaning.csv", index = True)

# clean the training data
train['cleaned_text'] = train['article'].progress_apply(cleaning)

# filter out articles with character count <= 250 or highlight ratio outside [0.05, 0.5]
train['char_len'] = train['cleaned_text'].str.len()
train['highlight_len'] = train['highlights'].str.len()
train['highlight_ratio'] = train['highlight_len'] / train['char_len']

# check the amount of samples violating each rule and the whole number of samples violating either of the rules
mask_len = train['char_len'] > 250
mask_ratio_min = train['highlight_ratio'] >= 0.05
mask_ratio_max = train['highlight_ratio'] <= 0.5
print("Original samples:", len(train))
print("Removed by char_len <= 250:", (~mask_len).sum())
print("Removed by highlight_ratio < 0.05:", (~mask_ratio_min).sum())
print("Removed by highlight_ratio > 0.5:", (~mask_ratio_max).sum())
combined_mask = mask_len & mask_ratio_min & mask_ratio_max
print("Remaining after all filters:", combined_mask.sum())
print("Total removed by all filters:", len(train) - combined_mask.sum())

# delete the samples
mask = ((train['char_len'] > 250) & (train['highlight_ratio'] >= 0.05) & (train['highlight_ratio'] <= 0.5))
train = train[mask].copy()

# plot the cleaned data
plt.hist(train['char_len'], bins = 50)
plt.title("Distribution of Article Character Lengths (After Cleaning)")
plt.savefig("../data/stats/char_len_hist_after_cleaning.png")
plt.close()

# save descriptive statistics for post-cleaned data to CSV for later analysis
post_cleaning = train[['char_len', 'highlight_len', 'highlight_ratio']].describe()
post_cleaning.to_csv("../data/stats/length_stats_post_cleaning.csv", index = True)

# define common bin edges for pre- / post-cleaning histograms to allow comparison
bins = np.linspace(min(pre_clean_char_len.min(), train['char_len'].min()), max(pre_clean_char_len.max(), train['char_len'].max()), 51)

# plot the overlay of both pre-cleaned and post-cleaned data
plt.hist(pre_clean_char_len, bins = bins, color = 'blue', alpha = 0.5, label = 'Pre-cleaning')
plt.hist(train['char_len'], bins = bins, color = 'red', alpha = 0.5, label = 'Post-cleaning')
plt.title("Word Length Distribution: Pre vs Post Cleaning")
plt.xlabel("Word Count")
plt.ylabel("Frequency")
plt.legend()
plt.savefig("../data/stats/char_len_hist_comparison.png")
plt.close()

# clean the validation data and delete outliers
val['cleaned_text'] = val['article'].progress_apply(cleaning)
val['char_len'] = val['cleaned_text'].str.split().progress_apply(len)
val['highlight_len'] = val['highlights'].str.split().progress_apply(len)
val['highlight_ratio'] = val['highlight_len'] / val['char_len']
mask = ((val['char_len'] > 250) & (val['highlight_ratio'] >= 0.05) & (val['highlight_ratio'] <= 0.5))
val = val[mask].copy()

# save the processed data (without ids)
train[['cleaned_text', 'highlights']].to_csv(os.path.join(processed_path, 'train.csv'), index = False)
val[['cleaned_text', 'highlights']].to_csv(os.path.join(processed_path, 'validation.csv'), index = False)
test[['article', 'highlights']].to_csv(os.path.join(processed_path, 'test.csv'), index = False)

# build and print frequences of all characters present in the articles
vocab_counter = build_vocab(train['cleaned_text'])
print("Vocabulary:", vocab_counter)
items = sorted(vocab_counter.items(), key = lambda x: x[1], reverse = True)
chars, freqs = zip(*items)
plt.figure(figsize = (20, 8))
plt.bar(range(len(chars)), freqs)
plt.xticks(range(len(chars)), chars)
plt.xlabel("Character")
plt.ylabel("Log Frequency")
plt.title("Character Frequencies (Log Scale)")
plt.tight_layout()
plt.savefig("../data/stats/char_frequencies.png")
plt.close()

# compute how many symbols were deleted during cleaning
stats = cleaning_stats(train["article"], train["cleaned_text"])
print(stats)

# compute how many patterns were deleted
before = count_patterns(train["article"])
after = count_patterns(train["cleaned_text"])
impact = {k: before[k] - after[k] for k in before}
print(impact)

# track how many characters were normalized
total_augmented_chars = 0
train['ascii_folded_text'] = train['article'].progress_apply(track_ascii_fold)
print("Total normalized chars (ASCII fold):", total_augmented_chars)
print('Percent of normalized characters (ASCII fold):', 100 * total_augmented_chars / train['article'].str.len().sum())
