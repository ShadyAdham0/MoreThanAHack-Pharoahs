"""Configuration for the possession pipeline.

Every number here is a football prior or a measurement convention, not a parameter fitted to the
public ground truth. The public five minutes are used only to check that the pipeline runs, that
the frame alignment is right, and that the official scorer accepts the file (see RULES.md).
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Config:
    # -- data -------------------------------------------------------------------------------
    fps: int = 25
    n_frames: int = 199_319
    n_slots: int = 46
    pitch: tuple = (105.3, 68.0)          # metres, from games.py / corrected tracking

    # -- ball -------------------------------------------------------------------------------
    # Linear interpolation is only trusted over a short occlusion. A ball hidden for a second is
    # still the same ball; one hidden for five seconds has been kicked somewhere.
    ball_fill_frames: int = 15
    ball_speed_smooth: int = 9

    # -- possession evidence ----------------------------------------------------------------
    # Distance scale in metres: a player two metres from the ball is a strong possessor, ten
    # metres away says little. 1.6 m is roughly "within playing distance".
    team_sigma_m: float = 1.6
    # Beyond this distance nobody is controlling the ball, so the geometry is uninformative.
    control_range_m: float = 9.0
    # A fast-moving ball is a pass in flight: proximity to a player stops meaning control until
    # somebody receives it. Damp the team evidence towards 50/50 at this speed.
    pass_speed_mps: float = 11.0
    evidence_weight: float = 1.0

    # -- dead-ball evidence -----------------------------------------------------------------
    # Out of the pitch by more than this many metres, sustained, is out of play.
    out_of_play_margin_m: float = 0.75
    # Ball speed below this is "stationary" for dead-ball purposes.
    still_speed_mps: float = 0.55
    # Mean speed of the tracked players below which play is probably stopped. The absolute value is
    # only a fallback: the pipeline uses a percentile of the match's own speed distribution (see
    # features.py), because the median of a match includes the stoppages being detected.
    live_mean_speed_mps: float = 2.2
    # Which percentile of this match's collective-speed distribution counts as live tempo. The
    # median includes the stoppages being detected and sits at a standing-player speed; the upper
    # quartile calls too much of the match "slow" and over-triggers D. The 55th percentile is the
    # live-play shoulder: just above the stopped frames, below the moving ones.
    tempo_percentile: float = 55.0
    dead_smooth_frames: int = 25            # 1 s: a stoppage is not one quiet frame
    # A football match is dead for roughly a fifth to a third of its frames (throw-ins, goal
    # kicks, free kicks, injuries). The bias is deliberately negative so the speed and tag
    # evidence carry the call instead of the prior alone.
    dead_bias: float = -3.2
    dead_out_weight: float = 4.0
    dead_still_weight: float = 2.6
    dead_slow_weight: float = 2.8
    # A provider-tagged restart is direct evidence that play is stopped, so it should stand on its
    # own: 4.0 against the -3.2 bias is P(dead) = 0.69 before any other cue. It is not a hard
    # constraint because a tag can start slightly before or after the actual stoppage.
    dead_tag_weight: float = 4.0
    # A set piece is dead from the moment it is tagged until the ball is actually moving again.
    set_piece_max_dead_s: float = 25.0
    # Optional weak team prior from the Smart Tagging `Team` group (log-odds). Off: phases are
    # not possessions, and the most useful restart (throw-in) is tagged N/A.
    tag_team_weight: float = 0.0
    # Set pieces and kick-offs do name the team that will restart, and the restart is exactly
    # where a possession begins. A modest log-odds boost, held for a couple of seconds, is enough
    # to break the 50/50 tie at a kick-off without overriding what the tracking says later.
    restart_team_weight: float = 0.8
    kickoff_team_weight: float = 1.8
    restart_team_hold_s: float = 2.0
    kickoff_team_hold_s: float = 3.0

    # -- sequence model ---------------------------------------------------------------------
    # Per-frame evidence is noisy: the nearest tracked player changes on roughly a one-second
    # scale, while possession changes on a ten-second scale. Average the log evidence first, then
    # let the transition matrix supply the persistence. Centred, so edges stay where they are.
    evidence_smooth_frames: int = 51
    # A possession lasts about 14 s in open play; the dead-ball mean is deliberately shorter
    # because stoppages are numerous and quick. Both are football priors, not measurements of this
    # match or of the public window.
    mean_possession_frames: int = 350
    mean_dead_frames: int = 250
    # Share of possession changes that pass straight to the other team rather than via D.
    direct_switch_share: float = 0.78

    # -- run cleanup ------------------------------------------------------------------------
    min_team_run: int = 6                   # 0.24 s -- the GT contains a 0.76 s possession
    min_dead_run: int = 10                  # 0.40 s

    # -- graded blocks ----------------------------------------------------------------------
    # Published in DATA.md; not labels. Frames outside are X.
    graded: tuple = ((11_185, 90_651), (110_808, 190_197))
    kickoffs: tuple = (11_185, 113_742)


DEFAULT = Config()
