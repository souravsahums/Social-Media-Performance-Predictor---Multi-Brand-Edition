# Test Cases — Social Media Performance Predictor

This document contains 20 diverse test cases covering various scenarios for the `/predict-simple` API endpoint. Each test includes the JSON payload, expected outcome, and rationale.

---

## Test Case 1: High-Performance Reel with Celebrity Collaboration

**Scenario:** A perfectly optimized Reel — right duration, celebrity collab, strong caption with CTA, posted at peak time.

```json
{
  "metadata_content": {
    "caption": "Who's ready to feel the thunder? ⚡ Drop your city in the comments and win a Thums Up hamper! Tag 3 friends who need this energy 🔥 #ThumsUp #ToofaniEnergy #Giveaway",
    "media_name": "reel",
    "duration": 22,
    "is_collaborated_post": true,
    "collaborators": ["virat.kohli"],
    "created_at": "2026-07-15T19:00:00"
  },
  "profile_stats": {
    "username": "thumsupofficial",
    "followers": 350000,
    "post_author_username": "virat.kohli"
  },
  "media": [{"type": "video", "summary": "Virat Kohli drinking Thums Up, dramatic slow motion, stadium background, brand logo prominent"}]
}
```

**Expected:** High engagement  
**Rationale:** Reel + collab + optimal duration (22s) + CTA + question + evening (7PM) + person + brand visible

---

## Test Case 2: Low-Performance Static Post (Minimal Effort)

**Scenario:** A plain product shot with no caption effort, posted at a dead time.

```json
{
  "metadata_content": {
    "caption": "New.",
    "media_name": "post",
    "duration": 0,
    "is_collaborated_post": false,
    "collaborators": [],
    "created_at": "2026-03-10T03:00:00"
  },
  "profile_stats": {
    "username": "pepsiindia",
    "followers": 1200000,
    "post_author_username": "pepsiindia"
  },
  "media": [{"type": "thumbnail", "summary": "product can on plain white background"}]
}
```

**Expected:** Low engagement  
**Rationale:** Static post + 1-word caption + 3AM posting + large audience penalty + no CTA + no hashtags

---

## Test Case 3: Average Branded Reel — Medium Territory

**Scenario:** A standard branded reel, moderate length, reasonable caption but nothing special.

```json
{
  "metadata_content": {
    "caption": "The taste that keeps you going through Monday blues. Refresh your mood with every sip 🍋 #Sprite #MondayMotivation #StayCool",
    "media_name": "reel",
    "duration": 45,
    "is_collaborated_post": false,
    "collaborators": [],
    "created_at": "2026-05-12T12:00:00"
  },
  "profile_stats": {
    "username": "sprite_india",
    "followers": 170000,
    "post_author_username": "sprite_india"
  },
  "media": [{"type": "video", "summary": "bottle pouring into glass with ice cubes, product focus, brand colors"}]
}
```

**Expected:** Medium engagement  
**Rationale:** Reel (+12) helps, but noon posting (no timing bonus), no collab, slightly long (45s gets only +3)

---

## Test Case 4: Album with Strong Educational Caption

**Scenario:** Multi-image carousel with detailed content and question CTA.

```json
{
  "metadata_content": {
    "caption": "5 things you didn't know about Red Bull 🤯\n\n1. Founded in 1987 in Austria\n2. Sold in 175 countries\n3. Sponsors 600+ athletes worldwide\n4. Owns 2 F1 teams\n5. Gives you wings since '97\n\nWhich fact surprised you? Tell us below! 👇\n\n#RedBull #DidYouKnow #EnergyDrink #F1 #Sports #GivesYouWings",
    "media_name": "album",
    "duration": 0,
    "is_collaborated_post": false,
    "collaborators": [],
    "created_at": "2026-08-20T18:30:00"
  },
  "profile_stats": {
    "username": "redbullindia",
    "followers": 1800000,
    "post_author_username": "redbullindia"
  },
  "media": [{"type": "thumbnail", "summary": "infographic with Red Bull facts, bold text on dark background"}]
}
```

**Expected:** Medium engagement  
**Rationale:** Album format (+3), good caption + question + CTA + evening, but large audience penalty + no video advantage

---

## Test Case 5: Edge Case — Empty Caption

**Scenario:** Reel with no caption text at all.

```json
{
  "metadata_content": {
    "caption": "",
    "media_name": "reel",
    "duration": 15,
    "is_collaborated_post": false,
    "collaborators": [],
    "created_at": "2026-06-01T20:00:00"
  },
  "profile_stats": {
    "username": "cocacola_india",
    "followers": 2500000,
    "post_author_username": "cocacola_india"
  },
  "media": [{"type": "video", "summary": "people dancing at a party with Coca-Cola bottles, colorful lights"}]
}
```

**Expected:** Medium engagement  
**Rationale:** Reel + optimal duration + evening + person in visual, but no caption (-2) + no CTA + large audience (-3)

---

## Test Case 6: Edge Case — Extremely Long Video (180s)

**Scenario:** 3-minute reel — well beyond optimal length.

```json
{
  "metadata_content": {
    "caption": "Full BTS of our latest campaign! From concept to final cut — watch the complete journey 🎬✨ Every frame tells a story. #BTS #CocaCola #MakingOf #ContentCreation #BrandCampaign",
    "media_name": "reel",
    "duration": 180,
    "is_collaborated_post": true,
    "collaborators": ["director_kumar", "dop_sharma"],
    "created_at": "2026-09-10T17:00:00"
  },
  "profile_stats": {
    "username": "cocacola_india",
    "followers": 2500000,
    "post_author_username": "director_kumar"
  },
  "media": [{"type": "video", "summary": "behind the scenes footage, camera equipment, studio setup, team working"}]
}
```

**Expected:** Medium engagement  
**Rationale:** Collab with 2 creators helps significantly, but 180s duration gets -5 penalty, large audience -3

---

## Test Case 7: Weekend UGC — Multiple Collaborators

**Scenario:** Fan-created content reshared with multiple creators on a weekend.

```json
{
  "metadata_content": {
    "caption": "When the fam comes together for Thums Up! 🤘⚡ Shot by our amazing community. Keep the Toofani content coming!\n\nTag us and use #ThumsUpFam for a chance to be featured!\n\n📸 @adventure_junkie @mumbai_explorer",
    "media_name": "reel",
    "duration": 28,
    "is_collaborated_post": true,
    "collaborators": ["adventure_junkie", "mumbai_explorer"],
    "created_at": "2026-10-05T19:30:00"
  },
  "profile_stats": {
    "username": "thumsupofficial",
    "followers": 350000,
    "post_author_username": "adventure_junkie"
  },
  "media": [{"type": "video", "summary": "group of young people doing adventure activities, wearing branded gear, mountain backdrop"}]
}
```

**Expected:** High engagement  
**Rationale:** Reel + optimal duration + 2 collabs (~+16) + UGC + evening + weekend + person + CTA

---

## Test Case 8: Edge Case — Unknown Brand (Not in Training Data)

**Scenario:** A brand the model has never seen.

```json
{
  "metadata_content": {
    "caption": "Introducing our brand new flavor — Mango Madness! Available now 🥭🔥 Who's trying it first?",
    "media_name": "reel",
    "duration": 20,
    "is_collaborated_post": false,
    "collaborators": [],
    "created_at": "2026-04-01T18:00:00"
  },
  "profile_stats": {
    "username": "fanta_india",
    "followers": 50000,
    "post_author_username": "fanta_india"
  },
  "media": [{"type": "video", "summary": "mango being sliced, juice splashing, product bottle reveal at end"}]
}
```

**Expected:** Medium-High engagement  
**Rationale:** Good format + duration + evening + question + smaller audience boost (+3), but no brand context

---

## Test Case 9: Edge Case — Zero Followers (New Account)

**Scenario:** Brand new account with zero followers.

```json
{
  "metadata_content": {
    "caption": "Welcome to our official page! 🎉 Follow us for exciting content and giveaways. Hit that follow button! 🚀",
    "media_name": "post",
    "duration": 0,
    "is_collaborated_post": false,
    "collaborators": [],
    "created_at": "2026-01-15T10:00:00"
  },
  "profile_stats": {
    "username": "newbrand_india",
    "followers": 0,
    "post_author_username": "newbrand_india"
  },
  "media": [{"type": "thumbnail", "summary": "welcome message with brand logo, colorful design"}]
}
```

**Expected:** Low-Medium engagement  
**Rationale:** Static post (-5), no brand context, no video, but CTA present and morning posting

---

## Test Case 10: Red Bull — Extreme Sports Niche Content

**Scenario:** Classic Red Bull extreme sports content.

```json
{
  "metadata_content": {
    "caption": "Gravity is just a suggestion 🪂\n\nWatch @skydiver_raj push beyond limits at 15,000 feet. Would you try this? Comment YES or NO!\n\n#RedBull #GivesYouWings #Skydiving #ExtremeSports #Adventure",
    "media_name": "reel",
    "duration": 18,
    "is_collaborated_post": true,
    "collaborators": ["skydiver_raj"],
    "created_at": "2026-11-20T18:00:00"
  },
  "profile_stats": {
    "username": "redbullindia",
    "followers": 1800000,
    "post_author_username": "skydiver_raj"
  },
  "media": [{"type": "video", "summary": "Person skydiving, GoPro angle, clouds visible, Red Bull logo on helmet"}]
}
```

**Expected:** High engagement  
**Rationale:** Perfect combination: reel + collab + optimal duration + CTA + question + person + brand + evening

---

## Test Case 11: Pepsi — Celebrity Campaign

**Scenario:** Big-budget celebrity-driven content.

```json
{
  "metadata_content": {
    "caption": "Jab passion meets swag, magic happens 💙⚡\n\n@ranveersingh knows what it takes. Swag se karenge sabka swagat!\n\nNew TVC dropping tomorrow. Stay tuned! 📺\n\n#Pepsi #SwagStepChallenge #RanveerSingh",
    "media_name": "reel",
    "duration": 32,
    "is_collaborated_post": true,
    "collaborators": ["ranveersingh"],
    "created_at": "2026-03-25T20:00:00"
  },
  "profile_stats": {
    "username": "pepsiindia",
    "followers": 1200000,
    "post_author_username": "ranveersingh"
  },
  "media": [{"type": "video", "summary": "Ranveer Singh dancing with Pepsi can, high production value, multiple camera angles"}]
}
```

**Expected:** High engagement  
**Rationale:** Celebrity collab + reel + evening + person + brand + good hashtags + emojis

---

## Test Case 12: Lazy Content (Bare Minimum Effort)

**Scenario:** Absolute minimum effort brand post.

```json
{
  "metadata_content": {
    "caption": "#ad #sponsored",
    "media_name": "post",
    "duration": 0,
    "is_collaborated_post": false,
    "collaborators": [],
    "created_at": "2026-02-14T14:00:00"
  },
  "profile_stats": {
    "username": "cocacola_india",
    "followers": 2500000,
    "post_author_username": "cocacola_india"
  },
  "media": [{"type": "thumbnail", "summary": "plain product packaging image, no people, white background"}]
}
```

**Expected:** Low engagement  
**Rationale:** Static post (-5) + minimal caption + afternoon (no bonus) + large audience (-3) + no CTA/question

---

## Test Case 13: Morning Motivational — Poll Format

**Scenario:** Interactive content posted in the morning.

```json
{
  "metadata_content": {
    "caption": "Rise and grind! 🌅💪\n\nEvery morning is a fresh start. What's your energy booster?\n\nA) Red Bull 🥫\nB) Coffee ☕\nC) Pure willpower 💪\n\nComment below! #MorningMotivation #RedBull #EnergyBoost #MondayMood",
    "media_name": "reel",
    "duration": 12,
    "is_collaborated_post": false,
    "collaborators": [],
    "created_at": "2026-06-22T07:00:00"
  },
  "profile_stats": {
    "username": "redbullindia",
    "followers": 1800000,
    "post_author_username": "redbullindia"
  },
  "media": [{"type": "video", "summary": "sunrise time-lapse, person stretching, motivational text overlay"}]
}
```

**Expected:** Medium engagement  
**Rationale:** Reel + question + CTA + emojis + person, but short duration (12s no bonus) + morning (only +1) + large audience

---

## Test Case 14: Coca-Cola — Diwali Festival Campaign

**Scenario:** Festival-themed content tapping into cultural moments.

```json
{
  "metadata_content": {
    "caption": "This Diwali, open happiness with everyone you love 🪔✨🎆\n\nShare this reel with someone who makes your celebrations brighter! Double tap if you're excited 🎉\n\n#CocaCola #OpenHappiness #Diwali2026 #FestivalVibes #DiwaliWithCoke",
    "media_name": "reel",
    "duration": 30,
    "is_collaborated_post": false,
    "collaborators": [],
    "created_at": "2026-10-29T19:00:00"
  },
  "profile_stats": {
    "username": "cocacola_india",
    "followers": 2500000,
    "post_author_username": "cocacola_india"
  },
  "media": [{"type": "video", "summary": "family gathering for Diwali dinner, diyas lit, Coca-Cola bottles on table, warm colors, people smiling"}]
}
```

**Expected:** Medium engagement  
**Rationale:** Reel + optimal duration + evening + CTA + person + brand, but no collab + large audience penalty

---

## Test Case 15: Spam Hashtags (Over-optimization)

**Scenario:** Excessive hashtag usage that might trigger platform penalties.

```json
{
  "metadata_content": {
    "caption": "Summer vibes 🌊 #pepsi #cola #summer #drink #refreshing #cold #ice #cool #party #fun #friends #beach #sun #hot #trending #viral #explore #fyp #foryou #instagram #reels #india #mumbai #delhi #bangalore",
    "media_name": "reel",
    "duration": 25,
    "is_collaborated_post": false,
    "collaborators": [],
    "created_at": "2026-05-20T18:00:00"
  },
  "profile_stats": {
    "username": "pepsiindia",
    "followers": 1200000,
    "post_author_username": "pepsiindia"
  },
  "media": [{"type": "video", "summary": "person at beach holding Pepsi can, sunset background"}]
}
```

**Expected:** Medium engagement  
**Rationale:** Reel + duration + evening + person, but excessive hashtags (-2) + large audience + spammy feel

---

## Test Case 16: Gen Z Humor / Meme Content

**Scenario:** Trendy meme-style content targeting younger demographic.

```json
{
  "metadata_content": {
    "caption": "POV: Your friend says 'garmi nahi lag rahi' in 45°C 😂💀\n\nTag that friend who's always in denial! We'll send them a Sprite 🧊\n\n#Sprite #GarmiKaAntidote #SummerMemes #GenZ #RelatableContent",
    "media_name": "reel",
    "duration": 15,
    "is_collaborated_post": false,
    "collaborators": [],
    "created_at": "2026-06-05T20:30:00"
  },
  "profile_stats": {
    "username": "sprite_india",
    "followers": 170000,
    "post_author_username": "sprite_india"
  },
  "media": [{"type": "video", "summary": "split screen meme format, person sweating then drinking Sprite, comedic expressions"}]
}
```

**Expected:** High engagement  
**Rationale:** Reel + optimal duration + evening + CTA + emojis + good caption + smaller audience boost + person

---

## Test Case 17: Edge Case — Very Short Duration (5s)

**Scenario:** Extremely short reel below the engagement sweet spot.

```json
{
  "metadata_content": {
    "caption": "Quick sip! ⚡",
    "media_name": "reel",
    "duration": 5,
    "is_collaborated_post": false,
    "collaborators": [],
    "created_at": "2026-07-10T15:00:00"
  },
  "profile_stats": {
    "username": "cocacola_india",
    "followers": 2500000,
    "post_author_username": "cocacola_india"
  },
  "media": [{"type": "video", "summary": "quick product shot, Coke being poured"}]
}
```

**Expected:** Low engagement  
**Rationale:** Very short duration (no bonus), minimal caption, afternoon, large audience, no CTA

---

## Test Case 18: Multi-Creator Mega Collaboration (4 Creators)

**Scenario:** Brand event with 4 major creators — tests collaboration scaling.

```json
{
  "metadata_content": {
    "caption": "When India's biggest creators come together for one mission — SPREAD THE ENERGY! ⚡🇮🇳\n\nWatch the full episode on YouTube (link in bio) 🎬\n\nWho's your favorite creator? Comment below! 👇\n\n@tech_burner @ashish.chanchlani @bhuvan.bam @kusha.kapila",
    "media_name": "reel",
    "duration": 28,
    "is_collaborated_post": true,
    "collaborators": ["tech_burner", "ashish.chanchlani", "bhuvan.bam", "kusha.kapila"],
    "created_at": "2026-12-01T19:00:00"
  },
  "profile_stats": {
    "username": "redbullindia",
    "followers": 1800000,
    "post_author_username": "tech_burner"
  },
  "media": [{"type": "video", "summary": "group of creators doing extreme challenges, Red Bull branding, high energy atmosphere"}]
}
```

**Expected:** High engagement  
**Rationale:** 4 collabs (~+19 cap) + reel + optimal duration + evening + CTA + question + person + brand

---

## Test Case 19: Nostalgic Content — Emotional Angle

**Scenario:** Emotionally-driven content tapping into nostalgia.

```json
{
  "metadata_content": {
    "caption": "Remember when Sundays meant this? 🏏☀️🍕\n\nGully cricket, friends, and chilled Coke. Some things never change.\n\nTag your gully cricket squad! 👇 #CocaCola #GullyCricket #Nostalgia #90sKids #SundayVibes",
    "media_name": "reel",
    "duration": 25,
    "is_collaborated_post": false,
    "collaborators": [],
    "created_at": "2026-08-18T11:00:00"
  },
  "profile_stats": {
    "username": "cocacola_india",
    "followers": 2500000,
    "post_author_username": "cocacola_india"
  },
  "media": [{"type": "video", "summary": "kids playing cricket in gully, vintage filter, Coca-Cola glass bottle, family watching from balcony"}]
}
```

**Expected:** Medium engagement  
**Rationale:** Reel + optimal duration + CTA + question + person + brand, but 11AM + no collab + large audience

---

## Test Case 20: Edge Case — All Zeros / Minimal Input (Robustness Test)

**Scenario:** Absolute minimum input to test system doesn't crash.

```json
{
  "metadata_content": {
    "caption": "",
    "media_name": "post",
    "duration": 0,
    "is_collaborated_post": false,
    "collaborators": [],
    "created_at": ""
  },
  "profile_stats": {
    "username": "unknown_brand",
    "followers": 0,
    "post_author_username": "unknown_brand"
  },
  "media": []
}
```

**Expected:** Low engagement  
**Rationale:** All negative signals — static post, no caption, no context, no media description

---

## Summary of Test Coverage

| # | Scenario | Key Variables Tested |
|---|----------|---------------------|
| 1 | Perfect reel + celebrity collab | Optimal everything |
| 2 | Minimal static post | Worst case baseline |
| 3 | Average branded reel | Middle ground |
| 4 | Album + long educational caption | Different format |
| 5 | Empty caption | Missing text features |
| 6 | Very long video (180s) | Duration penalty |
| 7 | UGC + multiple collabs + weekend | Community content |
| 8 | Unknown brand | Generalization |
| 9 | Zero followers | Data edge case |
| 10 | Red Bull extreme sports | Brand-niche alignment |
| 11 | Pepsi celebrity | Celebrity effect |
| 12 | Lazy content | Minimum effort detection |
| 13 | Morning motivational + poll | Time-of-day + interactivity |
| 14 | Festival campaign (Diwali) | Seasonal/cultural timing |
| 15 | Spam hashtags | Over-optimization penalty |
| 16 | Gen Z meme humor | Caption quality + relatability |
| 17 | Very short video (5s) | Below duration floor |
| 18 | 4-person mega collab | Collab scaling limit |
| 19 | Nostalgic/emotional content | Sentiment angle |
| 20 | All zeros / empty | System robustness |
