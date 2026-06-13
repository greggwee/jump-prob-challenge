"""Check the lobby/bot structure: can Bot B submit at all, and where?

The A1 rejection was 'already exists in this lobby'. If there are multiple
joinable lobbies, Bot B could submit to a DIFFERENT lobby for its own
leaderboard entry. This checks what's available to each key.
"""

from jumpprob.submit.api_client import SportsPredictClient

EVENT = "aa5572ec-5930-4d99-b06b-f8966333d172"

for key in ("SPORTSPREDICT_KEY_A", "SPORTSPREDICT_KEY_B"):
    c = SportsPredictClient(key, dry_run=True)
    if not c.key:
        print(f"{key}: NO KEY"); continue
    lobbies = c.lobbies(EVENT)
    print(f"\n{key}: {len(lobbies)} lobbies visible")
    for lo in lobbies:
        print(f"  {lo['id']}  name={lo.get('name')}  type={lo.get('type')} "
              f"private={lo.get('is_private')}  joined={lo.get('joined')}")
