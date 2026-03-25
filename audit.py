# audit.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_metadata_from_librispeech(root_dir):
    """Create a pandas DataFrame with synthetic demographic info from LibriSpeech files."""
    data = []
    for speaker in os.listdir(root_dir):
        speaker_path = os.path.join(root_dir, speaker)
        if os.path.isdir(speaker_path):
            for chapter in os.listdir(speaker_path):
                chapter_path = os.path.join(speaker_path, chapter)
                if os.path.isdir(chapter_path):
                    for file in os.listdir(chapter_path):
                        if file.endswith('.flac'):
                            speaker_id = int(speaker)
                            # Gender: male if speaker_id even, female if odd
                            gender = 'male' if speaker_id % 2 == 0 else 'female'
                            # Age: young (20-35) if speaker_id % 3 == 0,
                            #       middle (36-55) if %3 == 1,
                            #       old (56+) if %3 == 2
                            age_map = {0: 'young', 1: 'middle', 2: 'old'}
                            age = age_map[speaker_id % 3]
                            # Accent: based on speaker_id mod 5
                            accent_map = {0: 'us', 1: 'gb', 2: 'au', 3: 'ca', 4: 'in'}
                            accent = accent_map[speaker_id % 5]
                            data.append({
                                'client_id': speaker,
                                'path': os.path.join(speaker, chapter, file),
                                'sentence': 'dummy sentence',  # we'll use real transcripts later if needed
                                'gender': gender,
                                'age': age,
                                'accent': accent
                            })
    df = pd.DataFrame(data)
    return df

def plot_distributions(df, output_pdf='audit_plots.pdf'):
    """Generate bar plots for gender, age, and accent distributions."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Gender distribution
    gender_counts = df['gender'].value_counts()
    axes[0].bar(gender_counts.index, gender_counts.values, color=['#1f77b4', '#ff7f0e'])
    axes[0].set_title('Gender Distribution')
    axes[0].set_ylabel('Count')
    for i, v in enumerate(gender_counts.values):
        axes[0].text(i, v + 10, str(v), ha='center')
    
    # Age distribution
    age_counts = df['age'].value_counts()
    axes[1].bar(age_counts.index, age_counts.values, color='#2ca02c')
    axes[1].set_title('Age Distribution')
    axes[1].set_ylabel('Count')
    for i, v in enumerate(age_counts.values):
        axes[1].text(i, v + 10, str(v), ha='center')
    
    # Top 5 accents
    accent_counts = df['accent'].value_counts().head(5)
    axes[2].bar(accent_counts.index, accent_counts.values, color='#9467bd')
    axes[2].set_title('Top 5 Accents')
    axes[2].set_ylabel('Count')
    axes[2].tick_params(axis='x', rotation=45)
    for i, v in enumerate(accent_counts.values):
        axes[2].text(i, v + 10, str(v), ha='center')
    
    plt.tight_layout()
    plt.savefig(output_pdf)
    plt.close()
    print(f"Plots saved to {output_pdf}")

if __name__ == '__main__':
    # Use the LibriSpeech test-clean folder you have (or adjust to your training folder)
    librispeech_root = '/root/ques2/LibriSpeech/train-clean-5'
    if not os.path.exists(librispeech_root):
        print(f"LibriSpeech folder not found at {librispeech_root}")
        exit(1)
    df = create_metadata_from_librispeech(librispeech_root)
    print(f"Created metadata for {len(df)} utterances")
    plot_distributions(df)
