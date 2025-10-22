"""
Checkpoint Renamer Callback
Đổi tên checkpoint theo metric (accuracy, F1, etc.) để dễ nhận biết
"""
import os
import shutil
from transformers import TrainerCallback


class SimpleMetricCheckpointCallback(TrainerCallback):
    """
    Callback đơn giản để rename checkpoint theo metric value
    
    Example:
        - checkpoint-500 -> checkpoint-8723 (nếu F1 = 0.8723, multiply_by=10000)
        - checkpoint-1000 -> checkpoint-9145 (nếu F1 = 0.9145, multiply_by=10000)
        - checkpoint-500 -> checkpoint-87 (nếu accuracy = 0.8723, multiply_by=100)
    
    Args:
        metric_name: Tên metric để đặt tên (default: 'eval_f1')
        multiply_by: Nhân metric với giá trị này trước khi làm tròn 
                     (default: 10000 cho F1 với 4 chữ số, 100 cho accuracy với 2 chữ số)
    """
    
    def __init__(self, metric_name='eval_f1', multiply_by=10000):
        super().__init__()
        self.metric_name = metric_name
        self.multiply_by = multiply_by
        self.renamed_checkpoints = {}
    
    def on_save(self, args, state, control, **kwargs):
        """
        Được gọi sau khi checkpoint được lưu
        Rename checkpoint folder theo metric value
        """
        # Lấy metric value từ logs
        if not state.log_history:
            return
        
        # Tìm log entry gần nhất có metric này
        metric_value = None
        for log_entry in reversed(state.log_history):
            if self.metric_name in log_entry:
                metric_value = log_entry[self.metric_name]
                break
        
        if metric_value is None:
            return
        
        # Tính tên mới
        metric_int = int(metric_value * self.multiply_by)
        
        # Tìm checkpoint folder vừa được lưu
        # Checkpoint thường có tên: checkpoint-{step}
        checkpoint_folder = f"checkpoint-{state.global_step}"
        checkpoint_path = os.path.join(args.output_dir, checkpoint_folder)
        
        if not os.path.exists(checkpoint_path):
            return
        
        # Tên mới theo metric
        new_checkpoint_name = f"checkpoint-{metric_int}"
        new_checkpoint_path = os.path.join(args.output_dir, new_checkpoint_name)
        
        # Nếu folder mới đã tồn tại, xóa nó (checkpoint cũ với cùng metric)
        if os.path.exists(new_checkpoint_path):
            shutil.rmtree(new_checkpoint_path)
        
        # Rename
        try:
            os.rename(checkpoint_path, new_checkpoint_path)
            # Format hiển thị: 0.8753 → 87.53% cho multiply_by=10000
            # hoặc 0.87 → 87% cho multiply_by=100
            display_pct = metric_value * 100
            print(f"\n📁 Renamed: {checkpoint_folder} -> {new_checkpoint_name} ({self.metric_name}={display_pct:.2f}%)")
            
            # Lưu mapping để reference
            self.renamed_checkpoints[checkpoint_folder] = new_checkpoint_name
            
        except Exception as e:
            print(f"\n⚠️  Could not rename checkpoint: {e}")


class BestMetricCheckpointCallback(TrainerCallback):
    """
    Callback nâng cao hơn - chỉ giữ lại checkpoint với metric tốt nhất
    
    Args:
        metric_name: Tên metric để theo dõi
        greater_is_better: True nếu metric càng cao càng tốt (default: True)
        keep_all: Nếu False, chỉ giữ checkpoint tốt nhất (default: False)
    """
    
    def __init__(self, metric_name='eval_accuracy', greater_is_better=True, keep_all=False):
        super().__init__()
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        self.keep_all = keep_all
        self.best_metric = None
        self.best_checkpoint = None
    
    def on_save(self, args, state, control, **kwargs):
        """Theo dõi và giữ lại checkpoint tốt nhất"""
        # Lấy metric value
        if not state.log_history:
            return
        
        metric_value = None
        for log_entry in reversed(state.log_history):
            if self.metric_name in log_entry:
                metric_value = log_entry[self.metric_name]
                break
        
        if metric_value is None:
            return
        
        # Check nếu đây là checkpoint tốt nhất
        is_best = False
        if self.best_metric is None:
            is_best = True
        elif self.greater_is_better and metric_value > self.best_metric:
            is_best = True
        elif not self.greater_is_better and metric_value < self.best_metric:
            is_best = True
        
        checkpoint_folder = f"checkpoint-{state.global_step}"
        checkpoint_path = os.path.join(args.output_dir, checkpoint_folder)
        
        if is_best:
            # Cập nhật best metric
            self.best_metric = metric_value
            
            # Xóa checkpoint cũ nếu không giữ tất cả
            if not self.keep_all and self.best_checkpoint is not None:
                old_path = os.path.join(args.output_dir, self.best_checkpoint)
                if os.path.exists(old_path):
                    try:
                        shutil.rmtree(old_path)
                        print(f"\n🗑️  Removed old checkpoint: {self.best_checkpoint}")
                    except Exception as e:
                        print(f"\n⚠️  Could not remove old checkpoint: {e}")
            
            self.best_checkpoint = checkpoint_folder
            
            # Rename checkpoint tốt nhất
            best_name = f"checkpoint-best-{int(metric_value * 100)}"
            best_path = os.path.join(args.output_dir, best_name)
            
            if os.path.exists(best_path):
                shutil.rmtree(best_path)
            
            try:
                shutil.copytree(checkpoint_path, best_path)
                print(f"\n⭐ Best checkpoint: {best_name} ({self.metric_name}={metric_value:.4f})")
            except Exception as e:
                print(f"\n⚠️  Could not copy best checkpoint: {e}")
