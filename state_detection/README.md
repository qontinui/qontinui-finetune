# State Detection Fine-tuning

This directory contains tools and utilities for fine-tuning pre-trained state detection models on specific GUI types and applications.

## What is Fine-tuning for State Detection?

Fine-tuning adapts general state detection models to specific domains:

- **Pre-trained models**: Trained on diverse GUI types with general state patterns
- **Fine-tuned models**: Specialized for specific applications, UI frameworks, or domains

Fine-tuning is particularly useful when:
1. You have a limited dataset for a new application
2. The application has domain-specific states not in the general model
3. You want to optimize for specific UI frameworks or platforms
4. The state transition patterns differ from the general case

## How State Detection Fine-tuning Differs from Element Detection Fine-tuning

| Aspect | Element Detection | State Detection |
|--------|------------------|-----------------|
| **Transfer Unit** | Visual features (CNN/ViT weights) | Visual features + temporal patterns |
| **What Adapts** | Element appearance recognition | State identification + transition dynamics |
| **Fine-tuning Data** | Labeled screenshots with elements | Screenshot sequences with state labels |
| **Domain Shift** | Visual appearance of UI elements | State definitions + transition patterns |
| **Typical Strategy** | Last layer(s) + backbone fine-tuning | Temporal model + backbone fine-tuning |

## Fine-tuning Scenarios

### Scenario 1: New Application with Similar UI Patterns

**Example**: Fine-tuning for a new web application when pre-trained on other web apps

**Approach**:
- Keep backbone frozen (visual features transfer well)
- Fine-tune transition predictor (learn app-specific transitions)
- Add new state labels to vocabulary
- Requires: 20-50 labeled sequences

**Steps**:
```python
# Load pre-trained model
model = load_pretrained_model('state_detector_web_apps.pt')

# Freeze backbone
for param in model.backbone.parameters():
    param.requires_grad = False

# Fine-tune transition predictor
for param in model.transition_predictor.parameters():
    param.requires_grad = True

# Add new states
model.add_states(['checkout_page', 'payment_form', 'confirmation'])

# Fine-tune with low learning rate
fine_tune(model, new_data, lr=1e-5)
```

### Scenario 2: Different UI Framework

**Example**: Adapting from web apps to desktop applications (Qt, Electron)

**Approach**:
- Partial backbone fine-tuning (visual differences significant)
- Full fine-tuning of transition predictor
- Potentially different state granularity
- Requires: 100-200 labeled sequences

**Considerations**:
- Desktop apps may have different state patterns (menu bars, toolbars)
- Window management states (minimized, maximized, floating)
- Different transition speeds and patterns

### Scenario 3: Domain-Specific State Taxonomy

**Example**: Medical software with specialized states (patient view, diagnosis, treatment planning)

**Approach**:
- Keep general visual features
- Learn completely new state vocabulary
- Domain-specific transition patterns
- Requires: 50-100 labeled sequences per state type

**Special Requirements**:
- Hierarchical state modeling (main state → sub-states)
- Domain-specific constraints (certain transitions impossible)
- Custom evaluation metrics (domain accuracy)

### Scenario 4: Few-shot Adaptation

**Example**: Quick adaptation to a new app with only 5-10 sequences

**Approach**:
- Meta-learning or prototypical networks
- Leverage similarity to known states
- Minimal parameter updates
- Requires: 5-10 labeled sequences

**Techniques**:
- Prototypical networks: Learn state prototypes from few examples
- Meta-learning: MAML or similar for fast adaptation
- Transfer from most similar pre-trained model

## Fine-tuning Best Practices

### 1. Data Requirements

**Minimum Data for Fine-tuning**:
- **Similar domain**: 20-50 sequences (100-500 frames)
- **Different domain**: 100-200 sequences (500-2000 frames)
- **New state vocabulary**: At least 10 sequences per new state
- **Transition patterns**: At least 5 examples of each transition type

**Data Quality**:
- Cover all application states in training data
- Include diverse transition patterns
- Balance state representation (avoid heavily skewed distributions)
- Include edge cases and error states

### 2. Model Selection

**Choose pre-trained model based on**:
- Platform similarity (web, desktop, mobile)
- UI framework similarity
- State complexity (number of states)
- Transition dynamics (fast vs. slow, cyclic vs. hierarchical)

**Available pre-trained models** (TODO: Add actual models):
- `state_detector_web_general.pt`: General web applications
- `state_detector_desktop_general.pt`: Desktop applications
- `state_detector_mobile_general.pt`: Mobile applications
- `state_detector_electron.pt`: Electron apps
- `state_detector_qt.pt`: Qt applications

### 3. Fine-tuning Strategy

**Layer-wise Fine-tuning Recommendations**:

```python
# Conservative approach (very similar domains)
freeze: backbone (feature extractor)
train: transition_predictor, state_classifier
learning_rate: 1e-5
epochs: 10-20

# Moderate approach (somewhat different domains)
freeze: lower backbone layers (0-6)
train: upper backbone layers (7-12), transition_predictor, state_classifier
learning_rate: 5e-5 (backbone), 1e-4 (new layers)
epochs: 20-50

# Aggressive approach (very different domains)
freeze: none (full fine-tuning)
train: all layers
learning_rate: 1e-5 (backbone), 5e-5 (other layers)
epochs: 50-100
```

### 4. Hyperparameter Guidelines

**Learning Rates**:
- Pre-trained backbone: 1e-6 to 1e-5
- Transition predictor: 1e-5 to 1e-4
- New layers (if added): 1e-4 to 1e-3

**Batch Size**:
- Depends on GPU memory and sequence length
- Typical: 4-8 sequences per batch
- Can use gradient accumulation for larger effective batch sizes

**Sequence Length**:
- Match pre-training length if possible
- Can adapt with short transition sequences (3-5 frames)
- Longer sequences (10-20 frames) for complex transitions

**Data Augmentation**:
- Frame-level: color jittering, brightness/contrast
- Temporal: frame dropping, subsequence sampling
- Sequence-level: reverse sequences, concatenation

### 5. Evaluation

**Metrics for Fine-tuned Models**:
- **State classification accuracy**: Per-frame state prediction accuracy
- **Transition accuracy**: Correct prediction of next state
- **Transition matrix error**: Difference from ground truth transition probabilities
- **Domain-specific metrics**: Custom metrics for your application

**Validation Strategy**:
- Hold-out validation set from target domain
- Ensure all states and transitions represented
- Test on unseen sequences (not just unseen frames)
- Compare to pre-trained model baseline

## Expected Inputs

Fine-tuning requires:

1. **Pre-trained Model Checkpoint**:
   - Model weights (`.pt` or `.pth` file)
   - Model configuration
   - State vocabulary (if applicable)
   - Training metadata

2. **Target Domain Data**:
   - Screenshot sequences (see `qontinui-train/data/screenshot_sequences/README.md`)
   - State labels for new domain
   - Transition annotations
   - Optional: State region annotations

3. **Fine-tuning Configuration**:
   - Which layers to freeze/train
   - Learning rates per layer group
   - Data augmentation settings
   - Evaluation metrics

## Expected Outputs

Fine-tuned models produce:

1. **Adapted Model Checkpoint**:
   - Updated model weights
   - New state vocabulary (if changed)
   - Fine-tuning metadata (source model, target domain, etc.)

2. **Evaluation Results**:
   - Per-state accuracy
   - Transition prediction accuracy
   - Confusion matrices (states and transitions)
   - Attention visualizations

3. **Domain Adaptation Report**:
   - Which states transferred well
   - Which transitions needed most adaptation
   - Data efficiency metrics
   - Comparison to pre-trained baseline

## Example Use Cases

### Use Case 1: E-commerce Website Adaptation

**Scenario**: Adapt general web app model to specific e-commerce site

**States to add**:
- Product listing page
- Product detail page
- Shopping cart
- Checkout flow (multiple steps)
- Order confirmation

**Fine-tuning approach**:
```bash
python scripts/finetune_state_detector.py \
    --pretrained models/state_detector_web_general.pt \
    --data data/ecommerce_sequences/ \
    --freeze-backbone \
    --new-states "product_listing,product_detail,cart,checkout,confirmation" \
    --epochs 30 \
    --lr 1e-5
```

### Use Case 2: Medical Application (Different Domain)

**Scenario**: Adapt from general desktop apps to medical imaging software

**Challenges**:
- Specialized UI (image viewers, measurement tools)
- Complex state hierarchy (patient → study → series → image)
- Domain-specific transitions

**Fine-tuning approach**:
```bash
python scripts/finetune_state_detector.py \
    --pretrained models/state_detector_desktop_general.pt \
    --data data/medical_app_sequences/ \
    --train-backbone-layers 6-12 \
    --hierarchical-states \
    --epochs 100 \
    --lr-backbone 5e-6 \
    --lr-predictor 1e-4
```

### Use Case 3: Few-shot Adaptation

**Scenario**: Quick adaptation to new app with only 10 sequences

**Approach**:
```bash
python scripts/few_shot_adapt.py \
    --pretrained models/state_detector_web_general.pt \
    --support-sequences data/new_app/train/ \
    --query-sequences data/new_app/test/ \
    --method prototypical \
    --shots 5
```

## Tools and Scripts

### Available Scripts (TODO: Implement)

1. **`scripts/finetune_state_detector.py`**: Main fine-tuning script
2. **`scripts/evaluate_finetuned.py`**: Evaluate fine-tuned model
3. **`scripts/compare_models.py`**: Compare pre-trained vs. fine-tuned
4. **`scripts/few_shot_adapt.py`**: Few-shot adaptation
5. **`scripts/visualize_adaptations.py`**: Visualize what changed

### Configuration Templates

Example fine-tuning config (`configs/finetune_ecommerce.yaml`):

```yaml
pretrained:
  model_path: "models/state_detector_web_general.pt"
  load_optimizer: false

model:
  freeze_backbone: true
  add_states:
    - "product_listing"
    - "product_detail"
    - "shopping_cart"
    - "checkout_step1"
    - "checkout_step2"
    - "checkout_step3"
    - "order_confirmation"

data:
  train_sequences: "data/ecommerce/train/"
  val_sequences: "data/ecommerce/val/"
  sequence_length: 5
  augmentation:
    frame_dropout: 0.1
    color_jitter: true

training:
  epochs: 30
  batch_size: 8
  learning_rate: 1e-5
  optimizer: "adamw"
  weight_decay: 1e-5
  scheduler: "cosine"
  warmup_epochs: 3

evaluation:
  metrics:
    - "state_accuracy"
    - "transition_accuracy"
    - "per_state_f1"
    - "transition_matrix_error"
  visualize: true
  save_predictions: true
```

## Integration with qontinui-train

Fine-tuning builds on models from `qontinui-train`:

1. **Pre-training** (qontinui-train):
   - Train general models on diverse data
   - Learn universal visual features
   - Learn common state patterns

2. **Fine-tuning** (qontinui-finetune):
   - Adapt to specific applications
   - Learn domain-specific states
   - Optimize for target use case

3. **Deployment** (qontinui-api):
   - Serve fine-tuned models
   - Application-specific inference
   - Real-time state detection

## Troubleshooting

### Common Issues

**Problem**: Fine-tuned model performs worse than pre-trained
- **Solution**: You may be overfitting. Try freezing more layers, using more data augmentation, or reducing learning rate.

**Problem**: Some states never get predicted correctly
- **Solution**: Check class imbalance. May need to add more examples of rare states or use class weighting.

**Problem**: Transitions are predicted incorrectly
- **Solution**: Transition predictor may need more training. Try unfreezing it or training longer.

**Problem**: Model works on training domain but not target domain
- **Solution**: Domain gap is too large. May need more target domain data or domain adaptation techniques.

## Future Enhancements

- [ ] Meta-learning for few-shot adaptation
- [ ] Domain adaptation techniques (adversarial, MMD)
- [ ] Continual learning for evolving applications
- [ ] Transfer learning from related tasks (element detection → state detection)
- [ ] Multi-task fine-tuning (state detection + element detection)
- [ ] Automated hyperparameter tuning for fine-tuning

## References

- Transfer learning for computer vision
- Few-shot learning: Prototypical networks, MAML
- Domain adaptation: DANN, MMD
- Fine-tuning strategies: Universal Language Model Fine-tuning (ULMFiT)
