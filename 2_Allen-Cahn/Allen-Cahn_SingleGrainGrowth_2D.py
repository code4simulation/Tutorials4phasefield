import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

class PhaseFieldSimulation:
    """Phase field model 시뮬레이션 클래스"""

    def __init__(self, N=128, dx=1, dt=0.01, grad_coeff=0.1,
                 mobility=5, radius=5, pbc_x=True, pbc_y=True):
        """
        기초 파라미터 설정 및 시뮬레이션 구조 설정

        Parameters
        ----------
        N : int
            그리드 포인트 개수
        dx : float
            그리드 간격
        dt : float
            시간 스텝
        grad_coeff : float
            그래디언트 계수 (ε²)
        mobility : float
            이동성 (Mobility)
        radius : int
            초기 원형 영역 반경
        pbc_x : bool
            X 방향 주기 경계 조건
        pbc_y : bool
            Y 방향 주기 경계 조건
        """
        self.N = N
        self.dx = dx
        self.dt = dt
        self.grad_coeff = grad_coeff
        self.mobility = mobility
        self.radius = radius
        self.pbc_x = pbc_x
        self.pbc_y = pbc_y

        # 초기화
        self.phi = self._initialize_phi()
        self.step_count = 0

        # 저장된 상태들 (시뮬레이션 진행 중 일정 간격으로 저장)
        self.phi_history = []  # 저장된 phi 필드 리스트
        self.step_history = []  # 저장된 스텝 번호 리스트

    def _initialize_phi(self):
        """
        Order parameter phi 초기화

        Returns
        -------
        np.ndarray
            초기화된 phi 필드 (0으로 초기화, 중심에 반경 radius인 원형 영역을 1로 설정)
        """
        phi = np.zeros((self.N, self.N))
        x = np.arange(self.N)
        y = np.arange(self.N)
        X, Y = np.meshgrid(x, y)
        center = self.N // 2
        mask = (X - center)**2 + (Y - center)**2 <= self.radius**2
        phi[mask] = 1.0
        return phi

    def laplacian(self, phi):
        """
        라플라시안 계산 (Finite Difference Method)
        주기 경계 조건 지원

        Parameters
        ----------
        phi : np.ndarray
            Order parameter 필드

        Returns
        -------
        np.ndarray
            라플라시안 값
        """
        if self.pbc_x:
            phi_xp = np.roll(phi, -1, axis=0)
            phi_xm = np.roll(phi, 1, axis=0)
        else:
            phi_xp = np.zeros_like(phi)
            phi_xm = np.zeros_like(phi)
            phi_xp[:-1, :] = phi[1:, :]
            phi_xm[1:, :] = phi[:-1, :]

        if self.pbc_y:
            phi_yp = np.roll(phi, -1, axis=1)
            phi_ym = np.roll(phi, 1, axis=1)
        else:
            phi_yp = np.zeros_like(phi)
            phi_ym = np.zeros_like(phi)
            phi_yp[:, :-1] = phi[:, 1:]
            phi_ym[:, 1:] = phi[:, :-1]

        return (phi_xp + phi_xm + phi_yp + phi_ym - 4 * phi) / (self.dx * self.dx)

    def update_phi(self):
        """
        Order parameter phi 업데이트 (시간 스텝 한 번)

        Allen-Cahn 방정식: ∂φ/∂t = M[ε²∇²φ - (φ³ - φ)]
        여기서 M은 mobility, ε²는 grad_coeff
        """
        lap = self.laplacian(self.phi)
        self.phi += self.dt * self.mobility * (self.grad_coeff * lap - (self.phi**3 - self.phi))
        self.step_count += 1


    def run_simulation(self, nsteps, save_interval=None):
        """
        시뮬레이션 실행 및 상태 저장

        Parameters
        ----------
        nsteps : int
            시뮬레이션 스텝 수
        save_interval : int or None
            몇 스텝마다 phi를 저장할지. None이면 마지막 상태만 저장
        """
        self.phi_history = [self.phi.copy()]
        self.step_history = [self.step_count]

        for step in range(nsteps):
            self.update_phi()

            if save_interval is not None and self.step_count % save_interval == 0:
                self.phi_history.append(self.phi.copy())
                self.step_history.append(self.step_count)

        if save_interval is None or self.step_count % save_interval != 0:
            self.phi_history.append(self.phi.copy())
            self.step_history.append(self.step_count)

    def visualize_snapshots(self, num_snapshots=5):
        """
        저장된 시뮬레이션 과정을 동일한 크기의 정적 이미지로 표시하고,
        전체 Figure에 대한 공통 colorbar를 생성합니다.
        """
        if not self.phi_history:
            print("Error: No simulation history. Run simulation first.")
            return

        indices = np.linspace(0, len(self.phi_history) - 1, num_snapshots, dtype=int)

        fig, axes = plt.subplots(1, num_snapshots, figsize=(4 * num_snapshots, 4))

        if num_snapshots == 1:
            axes = [axes]

        im = None
        for idx, ax_idx in enumerate(indices):
            ax = axes[idx]
            phi_snap = self.phi_history[ax_idx]
            step_label = self.step_history[ax_idx]

            im = ax.pcolormesh(phi_snap, cmap='RdBu', vmin=-1, vmax=1)
            ax.set_title(f"Step {step_label}", fontsize=12)
            ax.set_aspect('equal')

        # Figure의 오른쪽에 colorbar를 위한 새로운 축(cax)을 직접 추가합니다.
        # [left, bottom, width, height] 값은 figure 전체 크기에 대한 상대적인 비율입니다.
        fig.subplots_adjust(right=0.9) # subplot들이 colorbar와 겹치지 않도록 조정
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) # 위치와 크기 조절

        if im is not None:
            fig.colorbar(im, cax=cbar_ax)

        plt.show()

    def visualize_animation(self, output_filename='phase_field_simulation.mp4', fps=10):
        """
        저장된 시뮬레이션을 mp4 animation으로 저장

        Parameters
        ----------
        output_filename : str
            저장할 mp4 파일명
        fps : int
            애니메이션 프레임 속도 (frames per second)
        """
        if not self.phi_history:
            print("Error: No simulation history. Run simulation first with run_simulation().")
            return

        fig, ax = plt.subplots(figsize=(6, 6))

        def animate(frame_idx):
            ax.clear()
            im = ax.pcolormesh(self.phi_history[frame_idx], cmap='RdBu', vmin=-1, vmax=1)
            ax.set_title(f"Time: {self.step_history[frame_idx]} steps")
            ax.set_aspect('equal')
            return [im]

        anim = FuncAnimation(fig, animate, frames=len(self.phi_history),
                           interval=100, blit=False, repeat=True)

        writer = FFMpegWriter(fps=fps)
        anim.save(output_filename, writer=writer)
        plt.close()
        print(f"Animation saved to {output_filename}")

if __name__ == "__main__":
    sim = PhaseFieldSimulation(
        N=128,
        dx=1,
        dt=0.01,
        grad_coeff=0.1,
        mobility=5,
        radius=5,
        pbc_x=True,
        pbc_y=True
    )

    # 시뮬레이션 실행 (100 스텝마다 저장)
    print("Running simulation and saving states...")
    nsteps = 2500
    sim.run_simulation(nsteps=nsteps, save_interval=100)

    # 방법 1: 스냅샷으로 시각화 (1×5 배열)
    print("Visualizing snapshots...")
    sim.visualize_snapshots(num_snapshots=5)

    # 방법 2: 애니메이션으로 저장 (mp4 형식)
    print("Generating animation...")
    sim.visualize_animation(output_filename='phase_field.mp4', fps=10)

    # 저장된 상태 개수 확인
    print(f"Total saved states: {len(sim.phi_history)}")
    print(f"Simulation steps: {sim.step_history}")