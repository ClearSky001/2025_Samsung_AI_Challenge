import argparse
import json
from pathlib import Path
from tqdm import tqdm

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def preprocess_vqav2(questions_path, annotations_path, image_dir, image_prefix, output_path):
    questions_data = load_json(questions_path)
    annotations_data = load_json(annotations_path)

    qid_to_question = {q['question_id']: (q['image_id'], q['question']) for q in questions_data['questions']}
    qid_to_answers = {
        ann['question_id']: ann['answers'][0]['answer']  # 대표 답변 하나만 사용
        for ann in annotations_data['annotations']
    }

    result = []
    for qid in tqdm(qid_to_question):
        if qid not in qid_to_answers:
            continue
        image_id, question = qid_to_question[qid]
        image_filename = f"{image_prefix}_{image_id:012d}.jpg"
        image_path = str(Path(image_dir) / image_filename)
        answer = qid_to_answers[qid]
        result.append({
            "image_path": image_path,
            "question": question,
            "answer": answer
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        
def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess VQAv2 dataset into JSON")
    parser.add_argument("--base_dir", default="../dataset/VQAv2", help="Base directory for VQAv2 dataset")
    parser.add_argument("--train_output", default="../dataset/VQAv2/train.json")
    parser.add_argument("--val_output", default="../dataset/VQAv2/val.json")
    return parser.parse_args()

if __name__ == "__main__":
    # 경로는 사용자 로컬 환경에 맞게 수정
    args = parse_args()
    base = Path(args.base_dir)
    
    # Train
    preprocess_vqav2(
        questions_path = base / "Train/questions/v2_OpenEnded_mscoco_train2014_questions.json",
        annotations_path = base / "Train/annotations/v2_mscoco_train2014_annotations.json",
        image_dir = base / "Train/train2014",
        image_prefix = "COCO_train2014",
        output_path = Path(args.train_output)
    )

    # Validation
    preprocess_vqav2(
        questions_path = base / "Validation/questions/v2_OpenEnded_mscoco_val2014_questions.json",
        annotations_path = base / "Validation/annotations/v2_mscoco_val2014_annotations.json",
        image_dir = base / "Validation/val2014",
        image_prefix = "COCO_val2014",
        output_path = Path(args.val_output)
    )
