class DetectConfig:
    def __init__(self):
        self.config = {
            'classes' : ['human', 'hard_hat', 'safety_vest'],
            'colors' : {
            'human': (255, 255, 255),  # 흰색
            'hard_hat': (0, 255, 0),   # 초록색
            'safety_vest': (0, 255, 255)  # 노란색
            },
            'thresholds' : {
            'human': 0.8,
            'hard_hat': 0.85,
            'safety_vest': 0.8
            }
        }
        
    def __getitem__(self,key):
        return self.config[key]
    
    def get_config(self):
        return self.config