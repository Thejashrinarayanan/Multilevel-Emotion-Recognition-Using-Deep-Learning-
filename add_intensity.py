import pandas as pd

# Load your dataset
df = pd.read_csv('data/eeg_emotions.csv')
print("✅ Original data loaded:", df.shape)

# Check the first few rows
print(df.head())

# Define thresholds (you can tweak these values)
def get_intensity(avg):
    if avg < 30:
        return "Low"
    elif 30 <= avg < 60:
        return "Medium"
    else:
        return "High"

# Apply intensity logic to each row
df['Intensity'] = df['Average'].apply(get_intensity)

# Save new dataset
df.to_csv('data/eeg_emotions_intensity.csv', index=False)
print("💾 Intensity levels added successfully!")
print(df.head())
