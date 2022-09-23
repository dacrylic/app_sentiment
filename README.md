# Mobile app sentiment miner
Trial program for app review sentiment mining

Main packages include :

1. google_play_scraper (https://pypi.org/project/google-play-scraper/)for scraping app reviews from google play store.

2. transformers (https://huggingface.co/docs/transformers/index) for sentiment classification of reviews
   1. pretrained model utilised was twitter-roberta-base-sentiment-latest by cardiffnlp (https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)

3. seaborn for visualization


Currently only supports Google App Store

#TO DO

Add Apple app store reviews to supplement review sentiment data
