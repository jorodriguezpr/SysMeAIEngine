# Sample documents

Two short, original documents for trying SysMeAIEngine end to end before
using your own files. They follow the **Learn → Start here** walkthrough in
the web UI (`/ui/help`).

| File | Language | Content |
|---|---|---|
| `el-templo-del-desierto.txt` | Spanish | A short story with names, places and numbers to ask about |
| `houseplant-care-guide.txt` | English | A practical how-to guide with specific facts |

## Try it

1. **Train from Folder** → Source folder: this `examples\sample-docs` folder,
   Job / model name: `sample-docs`, Chunking: `paragraph`. Lower **Max rows**
   (for example to 200) for a quick run, then click **Start**.
2. When the job is done, open **Models**, choose the corpus `sample-docs` for
   your answering model and click **Set**.
3. Ask in **Chat** (Grounded mode on).

## Test questions (answers are in the documents)

- ¿Cómo se llamaba el abuelo de Zelomi?  → Ithamar
- ¿Cuántos rollos contó Miriel?  → cuarenta y dos (42)
- ¿Por qué Zelomi no dejó que Assur se llevara los rollos?  → por una promesa a su abuelo
- How often should I repot a houseplant?  → every one to two years, when roots outgrow the pot
- Which plants are safe for cats and dogs?  → spider plant, calathea, parlor palm, Boston fern
- What causes brown, crispy leaf tips?  → dry air or inconsistent watering

## Out-of-documents question (should be refused)

- ¿Quién ganó el mundial de fútbol de 2010?

A polite "not in the documents" answer means grounding is working.
