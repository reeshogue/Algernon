import pyscreenshot
import pyautogui
from torchvision.transforms.functional import resize, to_tensor

def move_to_coords(coords):
	pyautogui.moveTo(coords[0]*pyautogui.size()[0], coords[1]*pyautogui.size()[1], duration=0)

def click(boolean):
	if boolean:
		pyautogui.click()

def get_shot(shot_size):
	shot = pyscreenshot.grab()
	shot = resize(shot, shot_size)
	shot = to_tensor(shot).unsqueeze(0)
	return shot

def get_action_shape():
	return (1, 1, 1)