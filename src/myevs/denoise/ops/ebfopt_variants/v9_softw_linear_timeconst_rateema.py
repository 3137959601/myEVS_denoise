# 涓嬪懆璁″垝锛?
from __future__ import annotations

"""EBF_optimized 鍙樹綋 V9锛氭妸鍏ㄥ眬 rate EMA 浠庘€滄寜鏇存柊娆℃暟鈥濇敼鎴愨€滄寜鐪熷疄鏃堕棿甯告暟鈥濄€?

鐩爣
----
鍦?V2锛坰oft-weight + 绾挎€?scale锛夌殑鍩虹涓婏紝淇濇寔 raw-score / weight / scale 鍏紡涓嶅彉锛?
鍙敼 **鍏ㄥ眬鍣０鐜?proxy锛坋vents per tick锛塃MA 鐨勬洿鏂版柟寮?*锛?

- 鏃э紙V2锛夛細姣忔鏇存柊閮界敤鍥哄畾 `rate_ema_alpha`
  - 绛夋晥鏃堕棿甯告暟渚濊禆浜嬩欢鍒拌揪鐜囷細浜嬩欢瓒婂瘑锛屽崟浣嶇湡瀹炴椂闂村唴鏇存柊娆℃暟瓒婂锛孍MA 瓒娾€滃揩鈥濄€?
- 鏂帮紙V9锛夛細鐢ㄧ湡瀹炴椂闂撮棿闅?`dt` 鎺ㄥ鍔ㄦ€佹洿鏂扮郴鏁帮紝浣?EMA 鍏锋湁鍥哄畾鏃堕棿甯告暟 `tau_rate`
  - `alpha(dt) = 1 - exp(-dt / tau_rate)`
  - dt 瓒婂ぇ鍒欐洿鏂拌秺鈥滆窡寰椾笂鈥濓紝dt 瓒婂皬鍒欏崟娆℃洿鏂版洿鈥滆皑鎱庘€濓紝鏁翠綋鐢?tau_rate 鎺у埗鍝嶅簲閫熷害銆?

杩欑被鍐欐硶甯歌浜庤繛缁椂闂翠竴闃朵綆閫氾紙RC锛夌鏁ｅ寲锛?
- 浣犵粰瀹氫竴涓兂瑕佺殑鈥滃搷搴旀椂闂粹€濓紙tau_rate锛夛紝涓嶇杈撳叆閲囨牱棰戠巼鎬庝箞鎶栵紝婊ゆ尝鍣ㄨ涓烘洿涓€鑷淬€?

閰嶇疆
----
閫氳繃鐜鍙橀噺璁剧疆鏃堕棿甯告暟锛堝崟浣?us锛夛細
- `MYEVS_EBFOPT_RATE_EMA_TAU_US`锛氶粯璁?100000锛?00ms锛?

澶囨敞
----
- `MYEVS_EBFOPT_RATE_EMA_ALPHA` 鍦ㄦ湰鍙樹綋涓笉浼氳鐢ㄤ綔涓绘帶鍙傛暟锛堜粛浼氳鍩虹被璇诲彇锛屼絾鏇存柊鏃朵笉鐢級銆?
- 璇ュ彉浣撹蛋 Python 瀹炵幇锛坣umba kernel 鏈鐩栵級銆?
"""

import math
import os

import numpy as np

from ._base import EbfOptVariantBase, scale_linear


class EbfOptV9SoftWLinearTimeConstRateEma(EbfOptVariantBase):
    """V9: soft-weight + linear scale + time-constant rate EMA."""

    variant_id: str = "softw_linear_timeconst_rateema"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 涓?V2 淇濇寔涓€鑷达紙鏂逛究瀵圭収瀹為獙锛?
        self._noise_weight_k: float = 4.0
        self._noise_weight_min: float = 0.02

        tau_us = int(os.environ.get("MYEVS_EBFOPT_RATE_EMA_TAU_US", "100000") or "100000")
        if tau_us <= 0:
            tau_us = 100000

        # 浣跨敤 tb 鎶婄湡瀹炴椂闂村父鏁版槧灏勫埌 tick锛堜笌杈撳叆娴佹椂闂村熀涓€鑷达級銆?
        tau_ticks = int(self.tb.us_to_ticks(int(tau_us)))
        self._rate_tau_ticks: int = max(1, int(tau_ticks))

    def _noise_weight(self, score_raw: float) -> float:
        s = float(score_raw)
        k = max(1e-6, float(self._noise_weight_k))
        w = 1.0 / (1.0 + (s / k))
        w = max(float(self._noise_weight_min), min(1.0, float(w)))
        return float(w)

    def _expected_noise_score_scale(self, *, tau_ticks: int, r: int) -> float:
        if tau_ticks <= 0 or r <= 0:
            return 1.0
        if self._ema_inv_dt <= 0.0:
            return 1.0

        w = int(self.dims.width)
        h = int(self.dims.height)
        area = max(1, w * h)

        neigh_px = int((2 * r + 1) ** 2 - 1)
        if neigh_px <= 0:
            return 1.0

        return float(
            scale_linear(
                ema_inv_dt=float(self._ema_inv_dt),
                area=area,
                tau_ticks=int(tau_ticks),
                neigh_px=int(neigh_px),
            )
        )

    def _update_global_rate(self, t: int, *, weight: float) -> None:
        """鎸夌湡瀹炴椂闂村父鏁版洿鏂板叏灞€鍣０鐜?EMA銆?

        绾﹀畾锛?
        - 杈撳叆娴佺殑鏃堕棿鍗曚綅鏄?tick锛堜笌 tb 涓€鑷达級
        - inv_dt = weight / dt 鏄姞鏉冨埌杈剧巼锛坋vents per tick锛変及璁?
        - alpha(dt) = 1 - exp(-dt / tau_rate_ticks)
        """

        last = self._last_t_global
        self._last_t_global = int(t)
        if last is None:
            return

        dt = int(t) - int(last)
        if dt <= 0:
            return

        w = float(weight)
        if not np.isfinite(w) or w <= 0.0:
            return

        inv_dt = w / float(dt)
        if self._ema_inv_dt <= 0.0:
            self._ema_inv_dt = float(inv_dt)
            return

        tau = float(self._rate_tau_ticks)
        # 鏁板€肩ǔ瀹氾細alpha = 1 - exp(-dt/tau) = -expm1(-dt/tau)
        x = float(dt) / tau
        a = -math.expm1(-x)
        if not np.isfinite(a):
            return
        a = max(1e-6, min(1.0, float(a)))

        self._ema_inv_dt = (1.0 - a) * float(self._ema_inv_dt) + a * float(inv_dt)
