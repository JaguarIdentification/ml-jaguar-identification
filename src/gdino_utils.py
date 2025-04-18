import fiftyone as fo


# Function to extract the best bbox for the head
def get_best_detection_head(detections):
    """
    Sort detections by box area (ascending) and confidence (descending).
    Returns the best detection (smallest box, highest confidence).
    """
    if detections:
        return sorted(
            detections,
            key=lambda d: (d.bounding_box[2] * d.bounding_box[3], -d.confidence)
        )[0]
    return None


# Function to process head bboxes
def select_best_detection_head(dataset, raw_bboxes_field_name):
    """
    Processes the dataset using only Strategy 1 (size + confidence).
    """
    for sample_idx, sample in enumerate(dataset):
        if sample_idx % 50 == 0:
            print(f"Processing sample {sample_idx}/{len(dataset)}")
        
        # Access raw detections
        detections = getattr(sample, raw_bboxes_field_name, fo.Detections()).detections
        
        # Get best detection using Strategy 1
        best_detection = get_best_detection_head(detections)
        
        if best_detection:
            sample["bboxes_head"] = fo.Detections(detections=[best_detection])
        else:
            sample["bboxes_head"] = fo.Detections()
            sample.tags.append("no_detection")
        
        sample.save()

    # Verification
    no_detection_count = len(dataset.match_tags("no_detection"))
    print(f"{no_detection_count}/{len(dataset)} samples have no detections")
    

# Function to process body bboxes
def select_best_detection_body(dataset, raw_bboxes_field_name):
    """
    Function to update each sample in the view by selecting the bounding box
    with the highest confidence, and ensuring a bounding box is detected for each sample.
    """
    # Get the bbox with the highest probability for each sample
    for sample_idx, sample in enumerate(dataset):
        if sample_idx % 50 == 0:
            print(f"Processing sample {sample_idx}/{len(dataset)}")
        # Get the detections stored in the "bboxes_head" field
        detections = sample[raw_bboxes_field_name].detections if sample[raw_bboxes_field_name] else []
        if detections:
            # Find the detection with the highest confidence
            best_detection = max(detections, key=lambda det: det.confidence)
            # Add new field with only the best detection
            sample["bboxes_body"] = fo.Detections(detections=[best_detection])
            sample.save()

    # Ensure that a bbox has been found for each sample
    for sample in dataset:
        if not sample[raw_bboxes_field_name]:
            print('No jaguar detected for sample:', sample)
    
