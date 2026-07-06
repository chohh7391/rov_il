# ROV 베이스 제어 설계 — DWTEK I90, 4-DOF locomanipulation

`template/mdp/rov_actions.py`의 6-DOF 속도 서보(`ROVVelocityAction`)가 orientation 명령 시
진동하는 문제를 계기로, "베이스를 어떻게 움직일 것인가"를 처음부터 다시 설계한 결과. 대상 차량을
**DWTEK Investigator 90(I90)** 로 확정하고, 그 하드웨어에 맞춘 제어 구조를 정의한다.

---

## 0. TL;DR

1. **아키텍처 = 2계층 locomanipulation.** 베이스는 저수준 속도추종 제어기(또는 RL 정책)가 구동하고,
   팔(manipulator)은 데모 수집 → Imitation Learning(IL)이 학습한다. **IL은 베이스를 소유하지 않는다.**
2. **베이스는 4-DOF 작동.** I90은 관측급(observation-class)이라 스러스터가 surge/sway/heave/yaw만
   작동시키고 **roll/pitch는 복원 모멘트로 수동 안정**된다. 명령 벡터는 `[vx, vy, vz, ψ̇]`.
3. **저수준 제어 = Fossen 피드포워드 + PID (wrench-레벨).** yaw는 heading-hold, heave는 depth-hold,
   수평은 속도제어, roll/pitch는 제어하지 않고 복원항으로 모델링만 한다.
4. **파라미터는 I90 카탈로그에서 확정** (비대칭 surge 588/294 N 등, 아래 §4.4).
5. 인터페이스(`[vx,vy,vz,ψ̇] → body wrench`)를 고정하면 나중에 RL `.pt`를 그대로 스왑 가능
   (IsaacLab `locomanipulation/pick_place`의 `AgileBasedLowerBodyAction` 방식).

---

## 0.5 구현 상태 (2026-07-06)

| 단계 | 상태 | 비고 |
| --- | --- | --- |
| **단계 1 — 4-DOF 제어 리팩터** | ✅ 구현 (`rov_actions.py`) | 선형 PI + **attitude = 원본 각속도 서보**(roll/pitch 목표 0 → 감쇠, yaw는 rate 추종) + 비대칭 추력 클램프. **heading-hold·level-hold 스프링은 되돌림** — attitude position 스프링은 translation이 팔 COM으로 유발한 미세 tilt/yaw에 링잉함. 자동 수평·heading-hold는 Stage 2(Fossen) 이후 sim 튜닝으로 이관. **sim 런타임 검증은 미완.** |
| **단계 3 — cfg 파라미터** | ✅ 구현 | I90 비대칭 추력한계(588/294·329·304 N), `lin_vel_scale`(1.5,1.0,0.7), yaw_rate 0.6 rad/s를 `ROVVelocityActionCfg` 기본값으로. |
| **단계 2 — Fossen FF** | ✅ 구현 (`template/mdp/hydrodynamics.py` + EventTerm) | MarineGym 수식을 순수함수 `compute_hydro_wrench`로 이식, `SingleArmEventCfg.apply_hydro` EventTerm(instantaneous+add)으로 배선. **안전 서브셋 기본 ON**: damping + 부력복원모멘트(→roll/pitch 자동수평); coriolis/added-mass/부력힘은 OFF 플래그. 계수는 BlueROV.yaml(현 USD와 일치). 순수함수 6개 sanity 통과(축플립·부호 OK). **sim 런타임 검증·리지드바디 damping 축소는 미완**(후속). |
| **단계 3 — 에셋(질량/부력)** | ⏸ 보류 | 질량은 USD에 baked → 120 kg 매칭은 USD 편집 필요. 리지드바디 damping은 Fossen 도입 전까지 유지(수동 복원 감쇠원). |

**인터페이스 결정:** `action_dim`은 6 유지(문서 §6의 대안). 키보드(14D)·`preprocess`(`==14`)·recorder·
lerobot 파이프라인을 깨지 않으려고, 6슬롯 중 roll/pitch(슬롯 3,4)는 제어에서 무시. 진짜 4슬롯으로 줄이려면
`rov_keyboard.py`/`action_process.py`를 함께 수정해야 함(후속).

---

## 1. 왜 다시 설계하나

기존 `ROVVelocityAction`은 6-DOF 바디 속도를 순수 P(각속도)로 서보한다
(`rov_actions.py:213` `torque_b = kp_angular * ang_vel_error`). 문제:

- **각속도 채널에 능동 감쇠(D/I)도 자세 홀드도 없음** (`rov_actions.py:209-213`) → 팔 반작용·복원
  모멘트·축간 커플링을 능동적으로 죽이지 못하고 링잉.
- **속도-only 제어는 heading을 유지 못 함** → 어떤 바이어스든 자세가 서서히 표류.
- 그런데 실제 목적은 **팔로 물체를 잡는 것**이고, 베이스는 "정지"가 아니라 "안정적으로 이동"해야 한다
  (locomanipulation).

→ 명령 primitive와 제어기 평형점을 하드웨어(I90)에 맞게 다시 잡는다.

---

## 2. 대상 차량: DWTEK I90 (control-relevant 스펙)

출처: `~/Downloads/I90_catalogue_rov.pdf` (Investigator 90, Observation Class).

| 항목 | 값 | 제어적 함의 |
| --- | --- | --- |
| 무게(공기 중) | 120 kg | sim 질량; 중성부력 가정 |
| 크기 (L×W×H) | 1,100 × 700 × 490 mm | yaw 모멘트암 산정(폭 0.7 m) |
| 최대속도 | 3 knots ≈ **1.54 m/s** | `lin_vel_scale` 상한 |
| 수평 스러스터 | **DWT6537 × 4** | surge + sway + yaw (벡터드 배치) |
| 수직 스러스터 | **DWT6537 × 2** | heave (+수동 롤/피치 안정 기여) |
| Forward/reverse 추력 | **60 / 30 kgf** (588 / 294 N) | **비대칭 surge** 한계 |
| Lateral 추력 | 33.5 kgf (329 N) | sway 한계; sway가 작동함을 확인 |
| Vertical 추력 | 31 kgf (304 N) | heave 한계 |
| Heading 센서 | **Standard** | heading-hold 기준 |
| Depth 센서 | **Standard** | depth-hold 기준 |
| DVL / INS / USBL / Altitude / Sonar | Optional | 바디속도 / 전역상태 / 위치 / 고도 / 장애물 |
| Automatic control | Optional | auto-heading·depth·station-keep = 옵션 기능 |
| Manipulator | 5-axis × 1 (Optional) | pick_stone의 팔 |

**결정적 관찰:** 카탈로그가 추력을 Forward/reverse·Lateral·Vertical **세 축만** 명시하고 roll/pitch
추력은 없다. → 6개 스러스터로 **작동하는 DOF는 4개(surge/sway/heave/yaw)**. roll/pitch는 부심(COB)을
무게중심(COG) 위에 두어 얻는 **metacentric 복원 모멘트로 수동 안정**된다. (1 kgf = 9.80665 N.)

---

## 3. 전체 아키텍처: 2계층 locomanipulation

참조 구현: IsaacLab `manager_based/locomanipulation/pick_place`
(`AgileBasedLowerBodyActionCfg` — frozen 하체 정책이 `[vx,vy,wz,hip_height]`를 받아 다리 관절 타깃 생성;
상체는 Pink IK로 데모 수집). 우리 구조에 매핑:

| 계층 | 역할 | 구동 주체 | 학습 대상? |
| --- | --- | --- | --- |
| **베이스 층** | 속도명령 → wrench 추종 (I90 4-DOF) | RL 정책 또는 제어기 | ✗ (exogenous) |
| **팔 층** | 5-axis IK 조작, 데모 수집 | teleop 데모 | ✓ IL이 팔만 학습 |

- 우리 `SingleArmActionsCfg`가 이미 `rov_action`(베이스) + `arm_action` + `gripper_action`으로
  분리돼 있어 **2계층 골격은 그대로**다 (`rov_single_arm_env_cfg.py:145-150`).
- G1은 하체가 관절 구동이라 정책이 관절 타깃을 내지만, **ROV 베이스는 wrench 구동**이라 저수준 층의
  출력이 **wrench**다. 그래서 ROV의 "하체 정책" 아날로그 = 속도명령→wrench 추종기 = `ROVVelocityAction`.
- **"IL이 base 정지 가정?" 오해 해소:** IL은 베이스를 제어하지 않는다. 베이스는 별도 층이 움직이고
  IL은 그 위에서 팔만 학습한다. 따라서 (a) 데모는 **베이스가 움직이는 상태에서** 수집해야 하고,
  (b) 팔 관측·액션은 **베이스-상대**여야 한다 (→ §5, 이미 충족).

---

## 4. 베이스 층 제어 설계 (4-DOF)

### 4.1 명령 인터페이스 (고정 — 스왑 가능)

```
입력 action: [vx, vy, vz, ψ̇]   (바디 프레임, 4-DOF)
출력:        body wrench → world → permanent_wrench_composer  (현 경로 유지, rov_actions.py:236)
```

roll/pitch(wx, wy)는 명령에서 제거한다. 이 인터페이스 뒤에 나중에 RL `.pt`를 끼우면 그대로 교체된다.

### 4.2 축별 제어 모드

| 축 | 모드 | 기준 센서 | 비고 |
| --- | --- | --- | --- |
| surge(x), sway(y) | 속도제어 | DVL (`dvl_vel`, body frame) | 명령 = 목표 바디 속도 |
| yaw(ψ) | **heading-hold** | Heading (Std) | ψ̇ 명령을 heading setpoint로 적분, 각도 오차 PD |
| heave(z) | **depth-hold** | Depth (Std) | vz 명령을 depth setpoint로 적분, 깊이 오차 PD (또는 vz 속도제어) |
| (옵션) altitude | 고도-hold | Altitude (Opt) | 해저 상대 고도 |
| roll, pitch | **제어 안 함** | — | Fossen 복원항으로 수동 안정 (COB>COG) |

heading/depth setpoint는 **월드 프레임**에서 정의(요각, 월드 z)해야 "제자리 유지"가 잘 정의된다.
sim에서는 `root_quat_w`의 yaw, `root_pos_w`의 z(또는 barometer)를 쓴다.

### 4.3 제어 법칙: Fossen 피드포워드 + PID

```
wrench = wrench_ff(Fossen) + wrench_fb(PID) + 팔반작용 보정
```

- **wrench_ff (모델 기반 피드포워드):** added-mass·Coriolis·damping·복원(restoring)을 Fossen 6-DOF로
  계산해 미리 상쇄 (`docs/marinegym-hydrodynamics-eventterm.md` §4의 `compute_hydro_wrench`). PID는
  잔차만 잡으면 되므로 게인↓·감쇠↑ 가능.
- **wrench_fb (PID):**
  - surge/sway/heave: 선형 PI (기존 구조 유지, `rov_actions.py:195-207`).
  - yaw: 각도 오차 PD (heading-hold) — 각속도 채널에 **능동 감쇠(D) 추가**(현재 순수 P의 링잉 제거).
- **팔반작용 보정:** 현재 힘에만 적용되는 COM couple(`rov_actions.py:230-234`)을 유지/확장. 팔이
  COM을 이동시켜 생기는 기생 토크를 상쇄.

**이중감쇠 주의** (`marinegym-...md` §5): Fossen damping을 넣으면 리지드바디 `linear/angular_damping`
(`blue_rov.py:17-18`, 현재 10.0)과 중복된다. → 리지드바디 damping을 **0으로 내리고** 감쇠를 Fossen으로
일원화. 중성부력 가정이므로 buoyancy 복원항은 roll/pitch 안정에만 쓰고 순수 상방력이 되지 않게 주의.

### 4.4 파라미터 (I90 카탈로그 확정값)

| 파라미터 | 값 | 현재 기본값 → 교체 |
| --- | --- | --- |
| 질량 | 120 kg (중성부력) | blue_rov 값 → I90 |
| force_limit surge | **+588 N / −294 N (비대칭)** | 대칭 (100,100,100) → 방향별 |
| force_limit sway | ±329 N | ↑ |
| force_limit heave | ±304 N | ↑ |
| torque_limit yaw | 수평추력 × 모멘트암(폭 0.7 m)으로 산정 (배분행렬 기반, 캘리브레이션 필요) | (100,100,100) |
| lin_vel_scale | surge ~1.5, sway/heave 더 작게 (m/s) | (5,5,5) → ~1.5 |

> `force_limit`을 현재의 대칭 `tuple[float,float,float]`(`rov_actions.py:289`)에서 **방향별(+/−) 또는
> per-axis min/max** 로 바꿔야 비대칭 surge를 표현할 수 있다.

---

## 5. 팔 층 & 관측 프레임 (감사 결과: 통과)

pick_stone의 조작 관측·액션은 **이미 베이스-상대/egocentric**이라 움직이는 베이스에 적합하다.
정책이 보는 관측(`rov_single_arm_env_cfg.py` PolicyCfg, `:165-207`):

| 관측 | 프레임 | 판정 |
| --- | --- | --- |
| `joint_pos/vel(_rel)`, `joint_pos_target` | 고유수용 | 프레임 불변 ✓ |
| `ee_frame_state` | **베이스-상대** (`subtract_frame_transforms`, leisaac `enhance/envs/mdp/observations.py:104`) | ✓ |
| `dvl_vel` | 바디 프레임 (`dvl.data.lin_vel_b`) | ✓ |
| `wrist`/`front`/`sonar` 이미지 | egocentric | ✓ |
| `barometer`/`dvl_depth` | 절대 깊이(월드 z) | 전역 기준(고도계 성격) OK |

**핵심: 정책에 월드-프레임 물체(cube) 자세가 없다.** cube는 egocentric 비전으로만 인지. arm 액션은
relative IK(델타, `rov_actions.py:324` `ROVRelativeIKAction`) → 베이스-상대 ✓.

주의사항:
- `last_action`은 베이스 명령도 포함 → 팔 정책이 베이스 의도를 보는 건 유리. 단 **학습 레이블**은 팔+
  그리퍼만 (LeRobot feature가 `SINGLE_ARM_JOINT_NAMES`로 한정, `rov_single_arm_env_cfg.py:268`) ✓.
- **데모는 베이스가 움직이는 상태에서** 수집 (정지 데모만 모으면 배포 시 분포 이동).
- 나중에 상태기반 물체 관측을 추가하면 반드시 베이스-상대(`subtract_frame_transforms`)로.

---

## 6. 코드 변경 계획 (구현 순서)

### 단계 1 — `rov_actions.py` 4-DOF 리팩터 (최소 변경, 진동 직격)
- `action_dim` 6 → **4** (`rov_actions.py:121`). 명령 파싱 `[vx,vy,vz, ψ̇]`.
- yaw 채널을 **heading-hold**로: ψ̇ 명령을 요각 setpoint로 적분, `root_quat_w` yaw와의 각도 오차 PD
  (각속도 D 감쇠 포함). roll/pitch 토크 명령 제거.
- heave를 **depth-hold**로 (또는 vz 속도제어 유지, 우선 속도제어로 시작 후 depth-hold 추가).
- `force_limit`을 비대칭/per-axis min-max로, `lin_vel_scale` 하향 (§4.4).
- teleop 경로 정합: `init_action_cfg`/`preprocess_device_action`
  (`rov_lab/devices/action_process.py`)가 6-DOF를 가정하는지 확인 후 4-DOF로.

### 단계 2 — Fossen 피드포워드 이식
- `marinegym-hydrodynamics-eventterm.md` §4의 `compute_hydro_wrench`를 ActionTerm 내부(또는
  EventTerm)에서 호출. 리지드바디 damping 0으로(§5).

### 단계 3 — I90 파라미터/에셋
- `blue_rov.py`의 질량·추력·복원을 I90 값으로 재설정하거나 I90 에셋 신규 생성. 제어 *구조*는 동일,
  *파라미터*만 교체.

---

## 7. 진동 원인과 이 설계가 없애는 방식

| 진동 원인 (기존 6-DOF 서보) | 4-DOF + 이 설계에서 |
| --- | --- |
| roll/pitch 펜듈럼·축간 커플링 | **명령 안 함 → 복원 모멘트가 수동 안정** (하드웨어가 처리) |
| 각속도 순수 P의 지연 링잉 | yaw에 **능동 D 감쇠 + heading-hold** |
| 팔 반작용 커플링 | COM couple 보정 + Fossen FF로 잔차만 PID가 처리 |
| COB-COM 복원 진자 | Fossen 복원항을 **모델링**하고 그 위에서 안정 |
| 릴리스 스냅 | 명령 소스가 매끄러운 정책/제어기 → 부차적 |

즉 4-DOF 전환만으로 최대 원인(roll/pitch 관련)이 제거되고, 남는 yaw는 능동 감쇠로 잡는다.

---

## 8. 충실도 로드맵

- **현재: wrench-레벨.** 4-DOF 바디 wrench를 카탈로그 한계로 직접 적용. IL 데모·고수준 제어에 충분.
- **나중(sim-to-real): 스러스터-레벨.** 4수평+2수직 기하로 **추력배분행렬(TAM)** + 개별 스러스터
  추력곡선/포화. 실제 달성가능 wrench는 박스가 아니라 폴리토프라 포화 시 축이 커플됨. 실 I90에 학습
  결과를 올릴 때 필요. 인터페이스는 동일하게 유지되므로 저수준만 교체.

---

## 9. 검증 방법

- **단위/정성:** 베이스만(팔 hold) 구동해 각 축 속도 스텝 명령에 오버슈트/링잉 없이 추종하는지, yaw
  heading-hold가 외란 후 원 헤딩으로 복귀하는지, roll/pitch가 팔을 움직여도 수평으로 복원되는지.
- **커플링:** 팔을 뻗은 채 surge 가속 시 pitch 튐이 COM 보정으로 억제되는지.
- **IL 적합성:** 베이스가 움직이는 상태로 데모를 수집해도 egocentric 관측으로 파지가 재현되는지.
- 이중계산 점검: 리지드바디 damping을 껐는지, 중성부력과 buoyancy 복원이 모순되지 않는지 (§4.3).

---

## 10. 참조 (file:line)

**우리 코드**
- 베이스 서보(개선 대상): `rov_lab/source/rov_lab/tasks/template/mdp/rov_actions.py`
  (`ROVVelocityAction` `:35`, 각속도 순수 P `:213`, COM couple `:230-234`, `action_dim` `:121`,
  `force_limit` `:289`, wrench 적용 `:236`)
- 액션/관측 배선: `rov_lab/source/rov_lab/tasks/template/rov_single_arm_env_cfg.py`
  (actions `:145-150`, PolicyCfg `:165-207`, decimation `:257`, feature joints `:268`)
- 리지드바디 damping/부력: `rov_lab/source/rov_lab/assets/robots/blue_rov.py:17-18`
- ee_frame_state(베이스-상대): `external_dependencies/leisaac/.../enhance/envs/mdp/observations.py:104`

**Fossen/유체동역학**
- 모델·이식·이중감쇠 주의: `docs/marinegym-hydrodynamics-eventterm.md` (§4 모델, §5 통합 주의)

**IsaacLab locomanipulation 참조**
- 저수준 정책 ActionTerm: `.../manager_based/locomanipulation/pick_place/mdp/actions.py`
  (`AgileBasedLowerBodyAction`, action `[vx,vy,wz,hip_height]` `:61`)
- 조립 예시: `.../locomanipulation/pick_place/locomanipulation_g1_env_cfg.py` (ActionsCfg `:94-110`)

**차량 스펙**
- `~/Downloads/I90_catalogue_rov.pdf` (Investigator 90, Observation Class)
