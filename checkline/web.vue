<template>
  <div id="app">
    <canvas ref='canvas' id='canvas' width='228' height='228'></canvas>

    <div class="container">
      <div class="btn-group">
        <button @click='predict'>Predict</button>
        <button @click='reset'>Reset</button>
      </div>

      <div>
        <h2>Result</h2>
        <p>
          {{result}}
        </p>
      </div>
    </div>
  </div>
</template>

<script>
/* eslint-disable no-console */
import KerasJS from 'keras-js'
import _ from 'lodash'
import kerasModel from 'file-loader!./model.bin'

const model = new KerasJS.Model({
  filepath: kerasModel,
  gpu: true
})

model
  .ready()
  .then(() => {
    console.log('model is ready')
  })

export default {
  data() {
    return {
      result: null,
      canvas: null,
      ctx: null
    }
  },

  mounted() {
    const canvas = this.canvas = this.$refs.canvas
    const ctx = this.ctx = canvas.getContext('2d')

    canvas.addEventListener('mousedown', ev => {
      const {pageX: px, pageY: py} = ev
      ctx.beginPath()
      ctx.moveTo(...xy(px, py))

      const handler = ev => {
        const {pageX: x, pageY: y} = ev
        ctx.lineTo(...xy(x, y))
        ctx.stroke()
      }

      canvas.addEventListener('mousemove', handler)

      canvas.addEventListener('mouseup', () => {
        canvas.removeEventListener('mousemove', handler)
      })

      function xy(x, y) {
        return [
          x - canvas.offsetLeft,
          y - canvas.offsetTop
        ]
      }
    })

    this.reset()
  },

  methods: {
    reset() {
      console.log('clear')
      this.result = null
      this.ctx.fillStyle = 'rgb(255,255,255, 255)'
      this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height)
      this.ctx.strokeRect(99, 99, 30, 30)
    },

    predict() {
      const imageData = this.ctx.getImageData(100, 100, 28, 28).data

      // console.log(imageData)

      const input = reduceData(imageData)

      model.predict({input})
        .then(({output}) => {
          console.log(output)
          const [n, p] = output
          this.result = p > n ? 'OK' : 'NG'
        })

      function reduceData(imageData) {
        const gray = _.chunk(imageData, 4).map(([r,g,b]) => {
          return (r * 299 + g * 587 + b * 114) / 1000
          // return (r + g + b) / 3
        })

        const inverse = gray.map(x => 1 - x / 255)

        console.log('inverse', inverse)

        return new Float32Array(inverse)
      }
    },
  }
}

</script>

<style>
html, body {
  margin: 0;
  padding: 0;
  text-align: center;
}

canvas {
  margin: 100px 0 50px;
  outline: 1px solid #aaa;
}

button {
  margin: 1em;
}
</style>