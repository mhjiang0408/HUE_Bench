import os
import random
import csv
import os
import sys
# 确保可以导入项目根目录下的模块
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
from collections import defaultdict

# Path to the base directory containing all author folders
base_dir = './gocomics_downloads'

# Get all author directories
author_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

# Dictionary to hold images for each author
author_images = defaultdict(list)

# Load all image paths
for author in author_dirs:
    author_path = os.path.join(base_dir, author)
    images = [f for f in os.listdir(author_path) if f.endswith('.jpg')]
    
    # Only include authors with at least 2 images (for reference and ground truth)
    if len(images) >= 2:
        for img in images:
            author_images[author].append(os.path.join(author_path, img))

# Filter out authors with insufficient images
valid_authors = [author for author, images in author_images.items() if len(images) >= 2]

# Generate 100 questions
questions = []
for i in range(100):
    # Select reference author
    reference_author = random.choice(valid_authors)
    
    # Select reference image and ground truth (different image from same author)
    reference_image = random.choice(author_images[reference_author])
    remaining_images = [img for img in author_images[reference_author] if img != reference_image]
    ground_truth = random.choice(remaining_images)
    
    # Select 3 distractor authors (different from reference author)
    other_authors = [a for a in valid_authors if a != reference_author]
    distractor_authors = random.sample(other_authors, 3)
    
    # Select 1 image from each distractor author
    distractors = [random.choice(author_images[author]) for author in distractor_authors]
    
    # Create choices (1 ground truth and 3 distractors)
    choices = [ground_truth] + distractors
    random.shuffle(choices)  # Randomize order of choices
    
    # Find the index of the ground truth in choices (0-based)
    correct_index = choices.index(ground_truth)
    
    # Create the question entry
    question = {
        'id': i + 1,
        'reference_image': reference_image,
        'choice_A': choices[0],
        'choice_B': choices[1],
        'choice_C': choices[2],
        'choice_D': choices[3],
        'correct_answer': ['A', 'B', 'C', 'D'][correct_index],
        'reference_author': reference_author,
        'ground_truth_author': reference_author,
        'distractor_authors': ','.join(distractor_authors)
    }
    
    questions.append(question)

# Write to CSV
csv_file = './Data/political_comic_questions.csv'
with open(csv_file, 'w', newline='', encoding='utf-8') as f:
    fieldnames = ['id', 'reference_image', 'choice_A', 'choice_B', 'choice_C', 'choice_D', 
                  'correct_answer', 'reference_author', 'ground_truth_author', 'distractor_authors']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    
    writer.writeheader()
    for question in questions:
        writer.writerow(question)

print(f"Generated {len(questions)} questions in {csv_file}") 