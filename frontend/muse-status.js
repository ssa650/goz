/** @param {{source?:string, startupSamples?:number, targetSamples?:number, goodChannels?:string[], qualityError?:string, cleanSeconds:number, targetSeconds:number}} muse */
export function museProgress(muse) {
  if (muse.source === 'mindmonitor') {
    const contacts = muse.goodChannels?.length ? ` · good contacts: ${muse.goodChannels.join(', ')}` : '';
    return `Mind Monitor · ${muse.startupSamples ?? 0} / ${muse.targetSamples ?? 10} usable readings${contacts}${muse.qualityError ? ' · ' + muse.qualityError : ''}`;
  }
  return `${muse.cleanSeconds.toFixed(1)} / ${muse.targetSeconds} seconds of clean EEG${muse.qualityError ? ' · ' + muse.qualityError : ''}`;
}
