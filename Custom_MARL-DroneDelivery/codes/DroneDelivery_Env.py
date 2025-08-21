import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
class DroneDeliveryMultiAgentEnv(gym.Env):
    def __init__(self,num_drones=2,stochastic=False):
        super(DroneDeliveryMultiAgentEnv,self).__init__()
        self.grid_size=6
        self.stochastic=stochastic
        self.num_drones=num_drones
        self.agent_ids=[f"drone_{i+1}" for i in range(num_drones)]
        self.action_space=spaces.Discrete(6)
        self.observation_space=spaces.Tuple((
            spaces.Discrete(self.grid_size),
            spaces.Discrete(self.grid_size),
            spaces.Discrete(2) ))
        self.rewards={ "pickup":25, "delivery": 100,"step":-1, "no_fly_zone":-100,"hit_border":-50, "invalid_target":-25, "collision":-10 }
        self.package_positions={ "package_1":[2,2], "package_2":[3,3] }
        self.dropoff_positions={ "package_1":[5,5], "package_2":[5,0]}
        self.max_steps=100
        self.no_fly_zones=[(1,1),(1,4),(2,3),(3,0),(3,5),(4,2),(4,4)]

        self.bg_img=mpimg.imread("images/background.png")
        self.drone_imgs={
            "drone_1":mpimg.imread("images/drone1.png"),
            "drone_2":mpimg.imread("images/drone2.png")
        }
        self.pickup_imgs={
            ("drone_1","package_1"): mpimg.imread("images/drone1_pickup1.png"),
            ("drone_1","package_2"): mpimg.imread("images/drone1_pickup2.png"),
            ("drone_2","package_1"): mpimg.imread("images/drone2_pickup1.png"),
            ("drone_2","package_2"): mpimg.imread("images/drone2_pickup2.png")
        }
        self.package_imgs={
            "package_1":mpimg.imread("images/package1.png"),
            "package_2":mpimg.imread("images/package2.png")
        }
        self.dropoff_imgs={
            "package_1":mpimg.imread("images/dropoff1.png"),
            "package_2":mpimg.imread("images/dropoff2.png")
        }
        self.obstacle_img=mpimg.imread("images/obstacle.png")
        self.shared_obstacles=set()
        self.reset()
    def reset(self):
        self.agent_positions={"drone_1":[0,0],"drone_2":[0,5]}
        self.carrying={aid: None for aid in self.agent_ids}
        self.delivered={aid: False for aid in self.agent_ids}
        self.package_picked={"package_1":False,"package_2":False}
        self.steps=0
        self.shared_obstacles=set()
        observations={
            aid: tuple(self.agent_positions[aid]+[0]) for aid in self.agent_ids
        }
        return observations,{}
    def step(self,actions):
        self.steps+=1
        observations,rewards,terminations,truncations={},{},{},{}
        occupied_positions=set(tuple(pos) for pos in self.agent_positions.values())
        updated_positions=dict(self.agent_positions)
        shared_reward=0
        for aid, action in actions.items():
            reward=self.rewards["step"]
            x,y=self.agent_positions[aid]
            new_pos=[x,y]
            if self.stochastic and np.random.rand()<0.1:
                action=np.random.choice([a for a in range(4) if a != action])
            moves={0:[x,y+1],1:[x,y-1],2:[x-1,y],3:[x+1,y]}
            if action in moves:
                intended_pos=moves[action]
                if tuple(intended_pos) in self.shared_obstacles:
                    action=np.random.choice([a for a in range(4) if tuple(moves.get(a,())) not in self.shared_obstacles])
                    intended_pos=moves.get(action,new_pos)
                new_pos=intended_pos
            if action==4:
                for pid,ppos in self.package_positions.items():
                    if [x, y]==ppos and not self.package_picked[pid]:
                        self.carrying[aid]=pid
                        self.package_picked[pid]=True
                        reward=self.rewards["pickup"]
                        break
                else:
                    reward=self.rewards["invalid_target"]
            elif action==5:
                carried=self.carrying[aid]
                if carried and [x,y]==self.dropoff_positions[carried]:
                    self.delivered[aid]=True
                    self.carrying[aid]=None
                    reward=self.rewards["delivery"]
                elif carried:
                    reward=self.rewards["invalid_target"]
            if not (0<=new_pos[0]<self.grid_size and 0<=new_pos[1]<self.grid_size):
                reward=self.rewards["hit_border"]
            elif tuple(new_pos) in self.no_fly_zones:
                reward=self.rewards["no_fly_zone"]
                self.shared_obstacles.add(tuple(new_pos))
            elif tuple(new_pos) in occupied_positions and new_pos!=self.agent_positions[aid]:
                reward=self.rewards["collision"]
            else:
                updated_positions[aid]=new_pos
            shared_reward+=reward
            self.agent_positions[aid]=updated_positions[aid]
            occupied_positions=set(tuple(pos) for pos in updated_positions.values())
            observations[aid]=tuple(self.agent_positions[aid]+[int(self.carrying[aid] is not None)])
            terminations[aid]=self.delivered[aid]
            truncations[aid]=self.steps>=self.max_steps
        rewards={aid: shared_reward for aid in self.agent_ids}
        if all(self.delivered.values()):
            terminations={aid: True for aid in self.agent_ids}
        info={"shared_obstacles": list(self.shared_obstacles)}
        return observations,rewards,terminations,truncations,info
    def get_centralized_state(self):
        return {
            "positions": self.agent_positions,
            "carrying": self.carrying,
            "delivered": self.delivered,
            "package_picked": self.package_picked,
            "no_fly_zones": self.no_fly_zones,
            "step": self.steps,
            "shared_obstacles": list(self.shared_obstacles)
        }
    def render(self,plot=True):
        fig,ax=plt.subplots(figsize=(6,6))
        ax.set_xlim(0,self.grid_size)
        ax.set_ylim(0,self.grid_size)
        ax.set_xticks(range(self.grid_size+1))
        ax.set_yticks(range(self.grid_size+1))
        ax.grid(True,zorder=1)
        ax.imshow(self.bg_img,extent=[0,self.grid_size,0,self.grid_size],zorder=0)
        def draw(img,pos):
            ax.imshow(img,extent=[pos[0],pos[0]+1,pos[1],pos[1]+1],zorder=3)
        for zone in self.no_fly_zones:
            draw(self.obstacle_img,zone)
        for pid, ppos in self.package_positions.items():
            if not self.package_picked[pid]:
                draw(self.package_imgs[pid],ppos)
        for pid, drop_pos in self.dropoff_positions.items():
            if pid in self.package_positions and self.package_picked[pid] and all(self.carrying[aid]!=pid for aid in self.agent_ids):
                draw(self.package_imgs[pid],drop_pos)
            else:
                draw(self.dropoff_imgs[pid],drop_pos)
        for aid in self.agent_ids:
            if not self.delivered[aid]:
                pos=self.agent_positions[aid]
                carried=self.carrying[aid]
                img=self.pickup_imgs.get((aid,carried),self.drone_imgs[aid]) if carried else self.drone_imgs[aid]
                draw(img,pos)
        ax.set_title(f"Steps:{self.steps}",fontsize=16)
        if plot:
            plt.show()
        else:
            plt.close(fig)