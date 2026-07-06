# MarineGym dynamics를 Isaac Lab EventTerm으로 외력 적용하기

MarineGym의 수중 동역학(hydrodynamics)을 Isaac Lab `EventTerm`을 통해 ROV 베이스에 외력으로
가하는 방법 정리. 결론부터: **힘 계산 수식만 순수 함수로 포팅**하고, 그 결과 wrench를 매 스텝
`instantaneous_wrench_composer`에 더하는 EventTerm을 등록하면 된다. MarineGym 클래스를 직접
import하지 않는다(다른 시뮬레이터 스택이라 불가).

---

## 1. TL;DR

1. MarineGym의 힘 계산부(순수 PyTorch 수식, `underwaterVehicle.py:203-277`)를 프로젝트 안에
   standalone 함수 `compute_hydro_wrench(...)`로 이식한다. MarineGym 클래스는 `omni.isaac.core`
   기반이라 Isaac Lab 런타임에서 import 불가.
2. 매 스텝 발화하는 EventTerm(`mode="interval"`, `is_global_time=True`,
   `interval_range_s=(0.0, 0.0)`)에서 ROV 상태를 읽어 hydro wrench(body frame)를 계산.
3. 그 wrench를 `asset.instantaneous_wrench_composer.add_forces_and_torques(..., is_global=False)`로
   더한다. 이 컴포저는 매 sim 스텝 끝에 자동 0으로 리셋되며, 서보(permanent) wrench와 자동 합산된다.

---

## 2. Isaac Lab 외력 메커니즘 — 두 개의 wrench 컴포저

`RigidObject`/`Articulation`은 외력 버퍼를 두 종류로 관리한다
(`external_dependencies/IsaacLab/source/isaaclab/isaaclab/assets/rigid_object/rigid_object.py`).

| 컴포저 | 수명 | 용도(docstring) | 리셋 시점 |
| --- | --- | --- | --- |
| `instantaneous_wrench_composer` | **현재 sim 스텝만 유효** | *"drag forces처럼 매번 바뀌는 힘"* | **매 `write_data_to_sim()` 끝에 자동 0** (`rigid_object.py:168`) |
| `permanent_wrench_composer` | 리셋 전까지 지속 | *"모터 thrust처럼 일정한 힘"* | env reset 시에만 (`rigid_object.py:132`) |

**적용 로직** (`write_data_to_sim`, `rigid_object.py:134-168`): 매 sim 스텝에서
instantaneous가 active면 **permanent를 instantaneous에 더한 뒤 합을 PhysX에 적용**하고,
instantaneous를 0으로 리셋한다. 즉 `permanent + instantaneous`가 자동 합산되어 나간다.

→ 결론: **수중 hydro(drag/added-mass/buoyancy/Coriolis)는 `instantaneous`에 넣는 것이 설계 의도**.
서보/thrust는 `permanent`. 둘은 자동으로 합쳐지므로 서로 덮어쓰지 않는다.

### 컴포저 메서드 의미 (`utils/wrench_composer.py`)

- `add_forces_and_torques(forces, torques, body_ids, env_ids, is_global=False)` — **누적**(+=).
  shape: `forces/torques = (num_envs, num_bodies, 3)`. `is_global=False` = **body(link) frame**.
- `set_forces_and_torques(...)` — **덮어쓰기**(=). rov_action 서보가 `permanent`에 이걸 쓴다.
- `reset(env_ids)` — 해당 버퍼 0으로.

주의: `permanent`에 `add`를 쓰면 매 스텝 누적되어 발산한다(리셋이 env-reset 때뿐이라).
그래서 매 스텝 바뀌는 hydro는 반드시 **자동 리셋되는 `instantaneous`** 를 써야 한다.

---

## 3. EventTerm 배선 — 매 스텝 발화

Isaac Lab EventManager의 `"interval"` 모드는 env step마다 타이머를 감산해, 시간이 소진되면 발화한다
(`managers/event_manager.py:208-232`). `interval_range_s=(0.0, 0.0)` + `is_global_time=True`이면
**매 env step마다 `env_ids=None`(전체 env)로 발화**한다.

```python
# tasks/.../mdp/events.py
from __future__ import annotations
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

def apply_marinegym_hydro(
    env, env_ids,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="rov_base"),
):
    asset: Articulation = env.scene[asset_cfg.name]
    body_id = asset_cfg.body_ids[0]

    # 1) 상태 읽기 (world frame) → body frame twist
    #    quat 규약 주의: Isaac Lab은 wxyz. MarineGym도 wxyz.
    quat_w   = asset.data.body_quat_w[:, body_id]          # (N,4) wxyz
    lin_v_w  = asset.data.body_lin_vel_w[:, body_id]       # (N,3)
    ang_v_w  = asset.data.body_ang_vel_w[:, body_id]       # (N,3)
    # world→body 회전은 isaaclab.utils.math.quat_rotate_inverse 사용

    # 2) hydro wrench 계산 (포팅한 순수 함수, body frame)
    force_b, torque_b = compute_hydro_wrench(...)          # (N,3), (N,3)

    # 3) instantaneous 컴포저에 누적 (매 스텝 자동 리셋됨)
    asset.instantaneous_wrench_composer.add_forces_and_torques(
        forces=force_b.unsqueeze(1),      # (N,1,3)
        torques=torque_b.unsqueeze(1),    # (N,1,3)
        body_ids=asset_cfg.body_ids,
        env_ids=env_ids,                  # interval+global이면 None(전체)
        is_global=False,                  # body frame
    )
```

```python
# env cfg의 EventCfg
from isaaclab.managers import EventTermCfg as EventTerm

hydro = EventTerm(
    func=mdp.apply_marinegym_hydro,
    mode="interval",
    interval_range_s=(0.0, 0.0),   # 매 스텝
    is_global_time=True,           # 전체 env 동시 (env_ids=None)
    params={"asset_cfg": SceneEntityCfg("robot", body_names="rov_base")},
)
```

내장 참고 함수: `apply_external_force_torque`(`isaaclab/envs/mdp/events.py:1010`)가 동일 패턴으로
`permanent_wrench_composer.set_forces_and_torques`를 호출한다(랜덤 외력용). 우리는 hydro라서
`instantaneous` + `add`를 쓰는 점만 다르다.

### 타이밍 주의 (decimation)

- `env.step()`은 `decimation`번 물리 substep을 돌리고, `"interval"` 이벤트는 **env step당 1회**
  (decimation 루프 밖) 발화한다. 반면 `instantaneous`는 **substep마다** 리셋된다.
- **현재 이 프로젝트는 `decimation = 1`**(`rov_single_arm_env_cfg.py:257`)이라 env step = 1 substep →
  EventTerm이 substep과 1:1 대응해 **문제없다**(1-step ZOH 지연만 존재, 무시 가능).
- **만약 `decimation > 1`로 바꾸면** EventTerm은 substep 중 1회만 hydro를 넣어 과소적용된다.
  그때는 아래 "대안"처럼 ActionTerm에서 substep마다 적용해야 한다.

---

## 4. MarineGym 힘 계산 이식

### 왜 import가 아니라 포팅인가
MarineGym은 **`omni.isaac.core`(레거시 Isaac Sim/OmniDrones 스택)** 기반이다
(`marinegym/views/__init__.py:29-38`). Isaac Lab 2.x가 아니므로 `UnderwaterVehicle`을 import하면
Isaac Sim 런타임 의존성이 딸려와 우리 env에서 쓸 수 없다. 그러나 힘 **계산 수식 자체는 순수
PyTorch**(Isaac 의존 없음)이므로 그 부분만 가져온다.

### 모델(Fossen 6-DOF, body frame) — `underwaterVehicle.py`
- **Added mass** `calculate_added_mass()` (`:250`): `M_A · v̇_b`, 대각 6×6.
- **Coriolis(added-mass)** `calculate_corilis()` (`:256`): `M_A·v`와 body 속도의 외적.
- **Damping(선형+2차)** `calculate_damping()` (`:238`): `(D_lin + D_quad·|v|)·v`,
  off-diagonal 커플링 [1,5],[2,4],[4,2],[5,1] 포함.
- **Buoyancy/복원** `calculate_buoyancy()` (`:266`): `ρ·g·V` (ρ=997, g=9.8)를 roll/pitch로 투영 +
  COB-COM 오프셋(`coBM`) 복원 토크. (무게 자체는 PhysX 몫.)
- **Currents** `flow_vels`/`set_flow_velocities()` (`:168,327`): 상대속도로 반영.
- **가속도 추정** `calculate_acc()` (`:229`): body 속도의 유한차분 + 저역통과(alpha=0.3).
  PhysX가 body 가속도를 안 주므로 added-mass용으로 수치 추정.
- 오케스트레이터 `apply_hydrodynamic_forces(flow_vels_w)` (`:203-227`) → `(force_b, torque_b)` 반환.

### 파라미터 (dataclass 아님, YAML)
`marinegym/robots/assets/usd/BlueROV/BlueROV.yaml`:
`volume=0.0113459`, `coBM=0.01`, `drag_coef=0.3`,
`hydro_coef.added_mass`(6), `.linear_damping`(6), `.quadratic_damping`(6).
`initialize()`(`:154-167`)에서 `torch.diag(...)`로 6×6 행렬화. `masses/inertias/volume`은
`RigidPrimView`에서 읽음 → 우리 쪽에선 config 상수 + `asset.data`로 대체.

### 포팅 시 반드시 보존할 것 (watch-outs)
1. **쿼터니언 wxyz**(w 먼저). MarineGym·Isaac Lab 모두 wxyz라 다행. (`utils/torch.py:120`)
2. **축 플립 `[...,[1,2,4,5]] *= -1`** (`:212-213, 222-223`): MarineGym ENU/FLU ↔ Fossen NED 변환.
   그대로 유지.
3. **부호** `hydro = -(added_mass + coriolis + damping)` (`:221`).
4. **적용점 = COM**: `apply_forces_and_torques_at_pos`를 position 없이 호출(COM 적용). COB-COM
   복원 모멘트는 `coBM`로 수식에 이미 반영. → 우리도 `positions` 없이 body frame으로 적용.
5. **added-mass의 유한차분 가속도**는 큰 `dt`에서 노이즈/불안정 가능(솔버 밖에서 added mass를
   다루는 근본적 한계). `prev_body_vels`/`prev_body_acc` 상태를 스텝 간 유지해야 함 → 이벤트 함수가
   env에 버퍼를 물고 있거나(예: `env._hydro_prev_vel`), added-mass 항을 생략/보수적 튜닝.

---

## 5. 우리 ROV와 통합 시 주의 (이중 모델링 금지)

현재 `ROVVelocityAction`과 blue_rov 세팅은 이미 **단순화된 수중 거동**을 담고 있어, MarineGym hydro를
그대로 얹으면 **이중 계산**이 된다. 정리:

- **선형/각 damping 중복**: blue_rov 리지드바디에 `linear_damping`/`angular_damping`이 이미 있고
  (`assets/robots/blue_rov.py`), 서보에도 속도 PI(감쇠성)가 있다. MarineGym `calculate_damping`을
  추가하면 감쇠가 중복된다. → MarineGym drag를 쓰려면 **리지드바디 damping을 0으로** 두고 감쇠를
  hydro로 일원화.
- **부력 vs disable_gravity**: 베이스는 `disable_gravity=True`(중성부력 가정). MarineGym buoyancy는
  "무게는 PhysX가, 부력은 별도"를 전제한다. disable_gravity인 채 buoyancy만 더하면 순수 상방력으로
  떠오른다. → buoyancy까지 쓰려면 **gravity를 켜고 무게+부력을 함께** 두거나, 중성부력 가정이면
  **buoyancy 항을 빼고** drag/added-mass/Coriolis만 사용.
- **서보 재튜닝**: 서보 게인(`kp_linear` 등)은 외부 drag 없이 튜닝됨. 실제 drag를 추가하면 서보가
  더 세게 밀어야 하므로(현실적) 거동이 바뀐다 → 게인 재확인 필요.
- **합산은 안전**: 서보(permanent) + hydro(instantaneous)는 `write_data_to_sim`에서 자동 합산되어
  덮어쓰기 충돌은 없다(§2). 물리적 이중 계산만 주의.

---

## 6. 대안: EventTerm 대신 ActionTerm (정밀/decimation>1)

hydro는 속도 의존이라 원칙적으로 **substep마다** 갱신이 이상적이다. MarineGym 자신도
`apply_action()`에서 매 스텝 적용한다(`underwaterVehicle.py:172-201`). Isaac Lab에서 substep 훅은
`ActionTerm.apply_actions()`뿐이다(EventTerm은 env step당 1회). 따라서:

- **decimation=1**(현재): EventTerm 경로로 충분.
- **decimation>1 또는 고정밀 필요**: 기존 `ROVVelocityAction.apply_actions()` 안에서
  서보 wrench 계산 직후 `instantaneous_wrench_composer.add_forces_and_torques(hydro)`를 호출하는 방식이
  더 정확(현 서보는 `permanent`에 `set` → hydro는 `instantaneous`에 `add`, 자동 합산). 이 경우
  EventTerm은 불필요.

즉 "EventTerm으로 외력"은 **모듈식/선언적 구성**이 장점이고 decimation=1에서 정확하다. 정밀도가
중요해지면 ActionTerm으로 옮기면 된다. 버퍼(`instantaneous`)와 수식은 동일하게 재사용된다.

---

## 7. 검증 방법

- **단위 검증**: 포팅한 `compute_hydro_wrench`를 알려진 입력(정지 시 0 근처, 등속 시 drag가 속도
  반대 방향)으로 유닛테스트. 축 플립/부호가 맞는지 정지-강하 케이스로 확인.
- **정성 검증(sim)**: EventTerm만 켠 채(서보 0) ROV를 초기속도로 밀면 drag로 감속·정지하는지,
  buoyancy 포함 시 중성/양성부력 거동이 의도대로인지 관찰.
- **이중계산 점검**: 리지드바디 damping을 끄고 hydro로 일원화했는지, disable_gravity와 buoyancy가
  모순되지 않는지 확인(§5).

---

## 8. 참조 (file:line)

**MarineGym**
- 힘 계산: `external_dependencies/MarineGym/marinegym/robots/drone/underwaterVehicle.py`
  - `apply_hydrodynamic_forces` `:203-227`, `calculate_damping` `:238`, `calculate_added_mass` `:250`,
    `calculate_corilis` `:256`, `calculate_buoyancy` `:266`, `calculate_acc` `:229`
  - sim 적용부(`apply_forces_and_torques_at_pos`) `:195-199`, 파라미터 `initialize` `:154-167`
- 파라미터 YAML: `marinegym/robots/assets/usd/BlueROV/BlueROV.yaml`
- 프레임워크 증거(omni.isaac.core): `marinegym/views/__init__.py:29-38`

**Isaac Lab**
- 두 컴포저 & write_data_to_sim: `.../assets/rigid_object/rigid_object.py:105-168`
- WrenchComposer(add/set/reset): `.../utils/wrench_composer.py:123, 221, 320`
- 내장 외력 이벤트: `.../envs/mdp/events.py:1010` (`apply_external_force_torque`)
- EventManager interval 모드: `.../managers/event_manager.py:208-232`
- EventTermCfg(interval_range_s, is_global_time): `.../managers/manager_term_cfg.py:251-288`

**우리 ROV**
- 서보(permanent wrench 사용): `rov_lab/source/rov_lab/tasks/template/mdp/rov_actions.py`
  (`permanent_wrench_composer.set_forces_and_torques`)
- decimation/dt: `rov_lab/source/rov_lab/tasks/template/rov_single_arm_env_cfg.py:257`
- 베이스 리지드바디 damping/disable_gravity: `rov_lab/source/rov_lab/assets/robots/blue_rov.py`
