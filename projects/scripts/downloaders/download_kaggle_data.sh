USER=$1
DATASET=$2

# Change to the desired directory where you want to download the dataset
cd ../data

# Use Kaggle CLI to download the dataset
kaggle datasets download -d $USER/$DATASET

# Unzip the downloaded file
echo 'Unzipping...'
unzip -q -o $DATASET.zip -d $DATASET

# Remove the zip file after extraction
rm $DATASET.zip

# Confirmation message
echo '🚀 Done!'