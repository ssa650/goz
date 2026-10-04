/** Reject stale points and points belonging to the previous clip at a swap.
 * @param {{session_id?:string,clip_id?:string,t:number,valid?:boolean,target?:string}|null|undefined} point
 * @param {string|null} sessionId @param {string|undefined} clipId @param {number} nowSeconds
 */
export function currentObservation(point, sessionId, clipId, nowSeconds) {
  return point && point.session_id === sessionId && point.clip_id === clipId &&
    nowSeconds-point.t >= -.1 && nowSeconds-point.t <= .5 ? point : null;
}
