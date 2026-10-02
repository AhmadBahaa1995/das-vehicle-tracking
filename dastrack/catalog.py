"""
Month-catalog statistics and figure (manuscript Figure 8, Table 4).

Reads the CSV written by :func:`dastrack.production.write_catalog`, whose
``time`` column is already in local time (JST).
"""
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from .plotting import PAPER_RC


def load_catalog(csv_path, date_start=None, date_end=None):
    """Load the catalog, optionally keeping only ``date_start..date_end`` (inclusive, local time)."""
    df = pd.read_csv(csv_path, parse_dates=['time'])
    if date_start and date_end:
        start = pd.Timestamp(date_start)
        end_excl = pd.Timestamp(date_end) + pd.Timedelta(days=1)
        df = df[(df['time'] >= start) & (df['time'] < end_excl)]
    return df.reset_index(drop=True)


def _hourly_departures(df):
    """Per hour-of-day: each day's % departure of that hour's count from the day's mean."""
    hourly = df.set_index('time')['speed'].resample('h').size()
    hdf = hourly.reset_index(name='count')
    hdf['date'] = hdf['time'].dt.date
    hdf['hour'] = hdf['time'].dt.hour
    day_mean = hdf.groupby('date')['count'].transform('mean')
    hdf['pct'] = (hdf['count'] - day_mean) / day_mean.replace(0, np.nan) * 100.0
    return [hdf.loc[hdf['hour'] == h, 'pct'].dropna().values for h in range(24)]


def _volume_speed_3h(df):
    """3-hour volume and mean speed, each smoothed by a centred 3-bin rolling mean."""
    di = df.set_index('time')
    vol = di.resample('3h').size().rolling(window=3, min_periods=1, center=True).mean()
    spd = di['speed'].abs().resample('3h').mean().rolling(window=3, min_periods=1, center=True).mean()
    vol, spd = vol.align(spd, join='inner')
    ok = vol.notna() & spd.notna()
    return vol[ok], spd[ok]


def summarize_catalog(df):
    """Summary numbers of the kind reported in Table 4 / the Results text."""
    right = df.loc[df['speed'] > 0, 'speed']
    left = df.loc[df['speed'] < 0, 'speed'].abs()
    daily = df.set_index('time').resample('D').size()
    weekend = daily.index.dayofweek >= 5
    medians = np.array([np.median(b) if len(b) else np.nan for b in _hourly_departures(df)])
    vol, spd = _volume_speed_3h(df)
    hours = (df['time'].max() - df['time'].min()).total_seconds() / 3600 if len(df) else 0
    return dict(
        vehicles=len(df),
        mean_rate_per_hour=len(df) / hours if hours else np.nan,
        mean_speed_right_kmh=right.mean(), mean_speed_left_kmh=left.mean(),
        n_right=len(right), n_left=len(left),
        weekday_daily_min=daily[~weekend].min(), weekday_daily_max=daily[~weekend].max(),
        weekend_daily_min=daily[weekend].min(), weekend_daily_max=daily[weekend].max(),
        diurnal_min_hour=int(np.nanargmin(medians)), diurnal_min_pct=float(np.nanmin(medians)),
        diurnal_max_hour=int(np.nanargmax(medians)), diurnal_max_pct=float(np.nanmax(medians)),
        mean_duration_sec=df['duration_sec'].mean(), mean_distance_m=df['distance_m'].mean(),
        volume_speed_pearson_r=float(vol.corr(spd)),
    )


def make_figure8(df, save_path=None, show=False):
    """
    Five-panel catalog figure: (a) speed distributions by direction, (b) daily
    counts, (c) hourly % departure from the daily average, (d) 3-hour smoothed
    volume and speed over time, (e) volume vs speed with linear fit.
    """
    with plt.rc_context(PAPER_RC):
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.2], hspace=0.35, wspace=0.25)
        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1])
        ax_c = fig.add_subplot(gs[0, 2])
        ax_d = fig.add_subplot(gs[1, 0:2])
        ax_e = fig.add_subplot(gs[1, 2])

        # (a) speed distributions by direction
        right = df.loc[df['speed'] > 0, 'speed']
        left = df.loc[df['speed'] < 0, 'speed'].abs()
        if not right.empty and not left.empty:
            bins = np.linspace(0, max(right.max(), left.max()) * 1.05, 40)
            ax_a.hist(left, bins=bins, color='#f87171', alpha=0.65,
                      label=f'← (−) mean {left.mean():.1f} km/h')
            ax_a.hist(right, bins=bins, color='#34d399', alpha=0.65,
                      label=f'→ (+) mean {right.mean():.1f} km/h')
            ax_a.legend(fontsize=9, loc='upper right')
        ax_a.set_xlabel('Speed (km/h)')
        ax_a.set_ylabel('Vehicles')

        # (b) daily counts
        daily = df.set_index('time').resample('D').size()
        weekend = daily.index.dayofweek >= 5
        ax_b.bar(daily.index, daily.values, color=np.where(weekend, '#f59e0b', '#38bdf8'), width=0.8)
        ax_b.set_xlabel('Date')
        ax_b.set_ylabel('Vehicles / day')
        ax_b.legend(handles=[Patch(facecolor='#38bdf8', label='Weekday'),
                             Patch(facecolor='#f59e0b', label='Weekend')], fontsize=9)
        for lab in ax_b.get_xticklabels():
            lab.set_rotation(45)
            lab.set_ha('right')

        # (c) diurnal variability
        box_data = _hourly_departures(df)
        bp = ax_c.boxplot(box_data, positions=range(24), widths=0.6, patch_artist=True,
                          showfliers=True, flierprops=dict(marker='o', ms=3, alpha=0.4,
                                                           markerfacecolor='#6b7280',
                                                           markeredgecolor='none'))
        for box in bp['boxes']:
            box.set(facecolor='#c4b5fd', edgecolor='#4c1d95', alpha=0.9)
        for el in bp['medians']:
            el.set(color='#4c1d95', lw=2)
        for el in bp['whiskers'] + bp['caps']:
            el.set(color='#4c1d95', lw=1)
        ax_c.axhline(0, color='gray', lw=0.8, ls='--', alpha=0.6, zorder=0)
        ax_c.set_xlabel('Hour of day (JST)')
        ax_c.set_ylabel('Departure from daily average (%)')
        # A tick at every box (sparse set_xticks after a patch_artist boxplot
        # mis-renders on some matplotlib versions); label every third.
        ax_c.set_xticks(range(24))
        ax_c.set_xticklabels([str(h) if h % 3 == 0 else '' for h in range(24)])
        ax_c.set_xlim(-1, 24)

        # (d) volume and speed over time
        vol, spd = _volume_speed_3h(df)
        r = vol.corr(spd)
        ax_s = ax_d.twinx()
        ax_d.plot(vol.index, vol.values, color='#10b981', lw=2.0, label='Volume (3 h)')
        ax_d.fill_between(vol.index, vol.values, color='#10b981', alpha=0.25)
        ax_s.plot(spd.index, spd.values, color='#ef4444', lw=2.0, label='Mean speed (3 h)')
        ax_d.set_xlabel('Date & time (JST)')
        ax_d.set_ylabel('Vehicles / 3 h', color='#059669')
        ax_s.set_ylabel('Speed (km/h)', color='#dc2626')
        ax_d.tick_params(axis='y', labelcolor='#059669')
        ax_s.tick_params(axis='y', labelcolor='#dc2626')
        h1, l1 = ax_d.get_legend_handles_labels()
        h2, l2 = ax_s.get_legend_handles_labels()
        ax_d.legend(h1 + h2 + [Patch(color='none')], l1 + l2 + [f'Pearson r: {r:.2f}'],
                    loc='upper left', fontsize=9)
        ax_d.xaxis.set_major_locator(mdates.DayLocator(interval=3))
        ax_d.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))
        ax_d.grid(True, ls='--', alpha=0.4)

        # (e) volume vs speed
        ax_e.scatter(vol, spd, color='#6366f1', alpha=0.5, edgecolor='white', linewidth=0.5, s=30)
        if len(vol) > 1:
            z = np.polyfit(vol, spd, 1)
            xt = np.linspace(vol.min(), vol.max(), 100)
            ax_e.plot(xt, np.poly1d(z)(xt), color='#111827', ls=':', lw=2)
            ax_e.text(0.95, 0.95, f"Pearson $r = {r:.2f}$\n$R^2 = {r ** 2:.2f}$",
                      transform=ax_e.transAxes, fontsize=10, va='top', ha='right',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#e5e7eb'))
        ax_e.set_xlabel('Volume (vehicles / 3 h)', color='#059669')
        ax_e.set_ylabel('Speed (km/h)', color='#dc2626')
        ax_e.grid(True, ls='--', alpha=0.4)

        for ax, lab in zip([ax_a, ax_b, ax_c, ax_d, ax_e], 'abcde'):
            ax.set_title(f'({lab})', loc='left', fontweight='bold')

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        if show:
            plt.show()
        else:
            plt.close(fig)
