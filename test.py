from imdb_sentiment_analysis import preprocess_text


def test_preprocess_text():
    input_text = "This is a sample text with some punctuation marks! 1234"
    expected_output = "sampl text punctuation marks"
    actual_output = preprocess_text(input_text)
    print(actual_output)

    assert actual_output == expected_output


test_preprocess_text()