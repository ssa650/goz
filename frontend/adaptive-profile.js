/** Profile values are policy weights, never measured attention shares. A scene
 * count means evaluated scenes, including those without usable sensor evidence.
 * @param {{profile:{clips:number,characters:Record<string,number>},clips:{profileChanges?:{key:string,before:number,after:number}[],changes?:{key:string,before:number,after:number}[]}[]}} session
 */
export function profilePresentation(session) {
  const changes=session.clips.flatMap(c=>c.profileChanges ?? c.changes ?? []).filter(c=>
    Number.isFinite(c.before) && Number.isFinite(c.after) && c.before !== c.after);
  const characterUpdated=changes.some(c=>c.key.startsWith('character:'));
  const deliveryUpdated=changes.some(c=>c.key==='pacing' || c.key==='dialogue' || c.key.startsWith('genre:'));
  const evaluated=Math.max(0,session.profile.clips || 0);
  return {
    scenes:`${evaluated} scene${evaluated===1 ? '' : 's'} evaluated`,
    characterHeading:characterUpdated ? 'Tentative profile weights' : 'Starting profile weights',
    characterNote:characterUpdated ? 'Weights updated from supported comparative gaze evidence; these are not measured attention percentages.' :
      'No supported comparative attention update recorded. Starting weights do not mean equal attention.',
    deliveryHeading:deliveryUpdated ? 'Tentative delivery adjustments' : 'Default delivery settings',
    noChanges:evaluated ? 'No evidence-supported preference change recorded; starting settings retained.' :
      'No viewing evidence yet; starting settings retained.',
    characterUpdated,deliveryUpdated
  };
}
