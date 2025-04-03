# rendeer.py
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import imageio
from math import sin, cos, radians
import sys

class BankFraudVisualizer:
    def __init__(self):
        pygame.init()
        self.display = (800, 800)
        pygame.display.set_mode(self.display, DOUBLEBUF|OPENGL)
        pygame.display.set_caption("Bank Fraud 3D Visualization - Auto-GIF")
        
        self.init_gl()
        
        self.grid_size = 5
        self.cell_size = 2.0 / self.grid_size
        self.camera = {
            'angle': 25,
            'distance': 5,
            'rotation': 0,
            'auto_rotate': True,
            'initial_rotation': None,
            'recording_started': False
        }
        self.frames = []
        
        # Scene elements
        self.agent = {
            'pos': np.array([1.0, 3.0]),
            'color': [0.0, 0.5, 1.0, 1.0],
            'size': 0.3,
            'pulse': 0.5
        }
        self.atm = {
            'pos': np.array([2.0, 2.0]),
            'color': [0.0, 0.8, 0.2, 1.0],
            'size': 0.4,
            'pulse': 0.0
        }
        self.fraudsters = [
            {'pos': np.array([0.0, 0.0]), 'color': [1.0, 0.0, 0.0, 1.0], 'pulse': 0.5},
            {'pos': np.array([4.0, 1.0]), 'color': [1.0, 0.0, 0.0, 1.0], 'pulse': 0.5},
            {'pos': np.array([3.0, 4.0]), 'color': [1.0, 0.5, 0.0, 1.0], 'pulse': 0.8}
        ]
        
        self.running = True
        self.clock = pygame.time.Clock()
        self.fps = 30
        self.recording_complete = False

    def init_gl(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, [5, 5, 5, 1])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [1, 1, 1, 1])
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glClearColor(0.1, 0.1, 0.2, 1)

    def update_camera(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (self.display[0]/self.display[1]), 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        if self.camera['auto_rotate']:
            # Initialize recording if not started
            if not self.camera['recording_started']:
                self.camera['initial_rotation'] = self.camera['rotation']
                self.camera['recording_started'] = True
            
            self.camera['rotation'] = (self.camera['rotation'] + 0.5) % 360
            
            # Check if we've completed a full rotation
            if self.camera['recording_started'] and not self.recording_complete:
                if abs(self.camera['rotation'] - self.camera['initial_rotation']) < 0.6:
                    if len(self.frames) > 0:  # Ensure we've captured at least some frames
                        self.recording_complete = True
                        self.save_gif()
                        print("GIF saved after one complete rotation!")
            
        glTranslatef(0, -0.5, -self.camera['distance'])
        glRotatef(self.camera['angle'], 1, 0, 0)
        glRotatef(self.camera['rotation'], 0, 1, 0)

    def draw_grid(self):
        glDisable(GL_LIGHTING)
        glBegin(GL_LINES)
        glColor3f(0.5, 0.5, 0.5)
        for x in range(self.grid_size + 1):
            glVertex3f(x * self.cell_size, 0, 0)
            glVertex3f(x * self.cell_size, self.grid_size * self.cell_size, 0)
        for y in range(self.grid_size + 1):
            glVertex3f(0, y * self.cell_size, 0)
            glVertex3f(self.grid_size * self.cell_size, y * self.cell_size, 0)
        glEnd()
        glEnable(GL_LIGHTING)

    def draw_element(self, pos, color, size, pulse=0.0, shape='sphere'):
        glPushMatrix()
        glTranslatef(
            pos[0] * self.cell_size + self.cell_size/2,
            pos[1] * self.cell_size + self.cell_size/2,
            0
        )
        
        if pulse > 0:
            pulse_factor = 1.0 + 0.1 * pulse * sin(pygame.time.get_ticks() * 0.005)
            size = size * pulse_factor
        
        glColor4fv(color)
        
        if shape == 'sphere':
            quad = gluNewQuadric()
            gluSphere(quad, self.cell_size * size, 32, 32)
        elif shape == 'cube':
            size = self.cell_size * size
            self.draw_cube(size)
        
        glPopMatrix()

    def draw_cube(self, size):
        s = size / 2.0
        vertices = [
            [s, -s, -s], [s, s, -s], [-s, s, -s], [-s, -s, -s],
            [s, -s, s], [s, s, s], [-s, s, s], [-s, -s, s]
        ]
        
        faces = [
            [0,1,2,3], [4,5,6,7], [1,5,6,2],
            [0,4,7,3], [0,1,5,4], [3,2,6,7]
        ]
        
        glBegin(GL_QUADS)
        for face in faces:
            for vertex in face:
                glVertex3fv(vertices[vertex])
        glEnd()

    def capture_frame(self):
        buffer = glReadPixels(0, 0, self.display[0], self.display[1], GL_RGB, GL_UNSIGNED_BYTE)
        frame = np.frombuffer(buffer, dtype=np.uint8).reshape(self.display[1], self.display[0], 3)
        self.frames.append(np.flip(frame, axis=0))

    def save_gif(self, filename="rotation.gif"):
        if not self.frames:
            print("No frames captured!")
            return
            
        # Reduce frames if too many to keep GIF size manageable
        if len(self.frames) > 100:
            skip = len(self.frames) // 100
            self.frames = self.frames[::skip]
            
        try:
            imageio.mimsave(filename, self.frames, fps=20)
            print(f"Saved rotation GIF as {filename}")
        except Exception as e:
            print(f"Error saving GIF: {e}")

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False

    def render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.update_camera()
        self.draw_grid()
        
        self.draw_element(self.atm['pos'], self.atm['color'], self.atm['size'], shape='cube')
        self.draw_element(self.agent['pos'], self.agent['color'], self.agent['size'], self.agent['pulse'])
        
        for fraudster in self.fraudsters:
            self.draw_element(fraudster['pos'], fraudster['color'], 0.3, fraudster['pulse'])
        
        # Capture frame if recording
        if self.camera['recording_started'] and not self.recording_complete:
            self.capture_frame()
        
        pygame.display.flip()
        self.clock.tick(self.fps)

    def run(self):
        try:
            print("Recording one full rotation...")
            while self.running:
                self.handle_events()
                self.render()
                
                if self.recording_complete:
                    # Optionally keep running after saving GIF
                    # break  # Uncomment to exit after saving
                    pass
                    
        finally:
            pygame.quit()

if __name__ == "__main__":
    visualizer = BankFraudVisualizer()
    visualizer.run()